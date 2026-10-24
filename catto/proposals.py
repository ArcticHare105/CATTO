"""Generate LR-condition candidates in an 11-dimensional DCT gain space.

RGB channels share one gain map. Frequencies below radius 0.12 are frozen
before RGB clipping. Candidate 0 is fixed LR; candidate 1 is working LR;
the remaining proposals are antithetic offsets in DCT parameter space.
"""
import math

import torch

from .dct_gain import GlobalDCTBandGain


DCT11_BAND_EDGES = (0.12, 0.17, 0.22, 0.28, 0.35, 0.43,
                    0.52, 0.62, 0.72, 0.82, 0.90)
DCT11_DIMENSIONS = 11
DCT_SMOOTH_WIDTH = 0.05
DCT_ALPHA_MAX = 0.35
DCT_RAW_STRENGTH_CLAMP = 24.0
DCT_RAW_BAND_CLAMP = 12.0


def dct11_search_metadata():
    return {
        "condition_search_dimensions": DCT11_DIMENSIONS,
        "condition_search_anchor": "fixed_lr_condition",
        "condition_search_center": "current_dct_parameters",
        "condition_search_parameters": ["global_gain"] + [
            f"band_gain_{low:.2f}_{high:.2f}"
            for low, high in zip(DCT11_BAND_EDGES[:-1], DCT11_BAND_EDGES[1:])
        ],
        "dct_band_edges": list(DCT11_BAND_EDGES),
        "dct_smooth_width": DCT_SMOOTH_WIDTH,
        "dct_alpha_max": DCT_ALPHA_MAX,
        "dct_preserve_low": True,
        "dct_shared_rgb_gain": True,
        "dct_raw_parameter_scales": [DCT_RAW_STRENGTH_CLAMP] + [DCT_RAW_BAND_CLAMP] * 10,
        "noise_scale_definition": "raw_dct_parameter_std_multiplier",
        "condition_search_population": "fixed_lr,working_lr,antithetic_dct_proposals",
    }


def clip_dct11_parameters(params):
    scales = params.new_full((DCT11_DIMENSIONS,), DCT_RAW_BAND_CLAMP)
    scales[0] = DCT_RAW_STRENGTH_CLAMP
    return torch.maximum(torch.minimum(params, scales), -scales)


class DCTLRCondition:
    """Cache one immutable LR's DCT; render conditions from absolute gains."""
    def __init__(self, condition):
        if condition.ndim != 4 or condition.shape[:2] != (1, 3):
            raise ValueError("DCT LR condition must be a single NCHW RGB tensor")
        self.condition = condition.detach().clone()
        self.controller = GlobalDCTBandGain(band_edges=DCT11_BAND_EDGES,
            smooth_width=DCT_SMOOTH_WIDTH, alpha_max=DCT_ALPHA_MAX, preserve_low=True)
        with torch.autocast(device_type=condition.device.type, enabled=False):
            self.coeffs = self.controller.dct2(self.condition.float())

    @torch.no_grad()
    def render(self, params):
        with torch.autocast(device_type=self.condition.device.type, enabled=False):
            params = clip_dct11_parameters(params.float())
            shape = (params.shape[0],) + tuple(self.condition.shape[1:])
            gain = self.controller.build_gain_map(shape, params)
            proposals = self.controller.idct2(self.coeffs * gain).clamp(0, 1)
        proposals = proposals.to(self.condition.dtype)
        # Zero parameters are the exact original LR, avoiding DCT roundoff.
        return torch.where((params == 0).all(dim=1).view(-1, 1, 1, 1),
                           self.condition, proposals).detach()


def sample_dct11_parameters(count, noise_scale, device, generator=None):
    """Sample only 11 values per direction; interleave +p/-p and an odd tail."""
    if count < 0:
        raise ValueError("proposal count must be non-negative")
    noise_scale = float(noise_scale)
    if not math.isfinite(noise_scale) or noise_scale < 0:
        raise ValueError("DCT noise_scale must be finite and non-negative")
    half = torch.randn(count // 2, DCT11_DIMENSIONS, device=device,
                       dtype=torch.float32, generator=generator)
    params = torch.stack((half, -half), dim=1).reshape(-1, DCT11_DIMENSIONS)
    if count % 2:
        params = torch.cat((params, torch.randn(1, DCT11_DIMENSIONS, device=device,
                            dtype=torch.float32, generator=generator)), dim=0)
    scales = params.new_full((DCT11_DIMENSIONS,), DCT_RAW_BAND_CLAMP)
    scales[0] = DCT_RAW_STRENGTH_CLAMP
    params = params * (noise_scale * scales)
    return clip_dct11_parameters(params)

