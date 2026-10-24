"""DCT band-gain controller copied from seesr_antithetic_balanced_minimal.

The original global/band gain meanings and low-frequency freeze are retained.
See catto/proposals.py for the DCT11 configuration.
"""
from typing import Dict, Iterable, Tuple, Union

import torch
import torch.nn as nn


DEFAULT_DCT_BAND_EDGES = (0.12, 0.22, 0.38, 0.60, 0.85)


def normalize_band_edges(
    band_edges: Union[str, Iterable[float], torch.Tensor, None] = None,
) -> Tuple[float, ...]:
    if band_edges is None:
        values = DEFAULT_DCT_BAND_EDGES
    elif isinstance(band_edges, str):
        values = tuple(float(item.strip()) for item in band_edges.split(',') if item.strip())
    elif isinstance(band_edges, torch.Tensor):
        values = tuple(float(item) for item in band_edges.detach().cpu().flatten().tolist())
    else:
        values = tuple(float(item) for item in band_edges)

    if len(values) < 2:
        raise ValueError(f'dct_band_edges must contain at least 2 values, got {len(values)}.')
    if any(values[idx] >= values[idx + 1] for idx in range(len(values) - 1)):
        raise ValueError(f'dct_band_edges must be strictly increasing, got {values}.')
    if values[0] < 0.0 or values[-1] > 2.0 ** 0.5:
        raise ValueError(f'dct_band_edges must lie in normalized frequency radius, got {values}.')
    return values


class GlobalDCTBandGain(nn.Module):
    _basis_cache: Dict[Tuple[int, str, torch.dtype], torch.Tensor] = {}
    _basis_pair_cache: Dict[Tuple[int, int, str, torch.dtype], Tuple[torch.Tensor, torch.Tensor]] = {}
    _mask_cache: Dict[
        Tuple[int, int, str, torch.dtype, Tuple[float, ...], float],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ] = {}

    def __init__(
        self,
        band_edges: Union[str, Iterable[float], torch.Tensor, None] = None,
        smooth_width: float = 0.0,
        alpha_max: float = 1.0,
        preserve_low: bool = True,
        active_bands: Union[int, None] = None,
    ):
        super().__init__()
        self.band_edges = normalize_band_edges(band_edges)
        self.smooth_width = float(smooth_width)
        self.alpha_max = float(alpha_max)
        self.preserve_low = bool(preserve_low)
        self.num_bands = len(self.band_edges) - 1
        self.param_dim = 1 + self.num_bands
        requested_active_bands = self.num_bands if active_bands is None else int(active_bands)

        if self.smooth_width < 0.0:
            raise ValueError('dct_smooth_width must be non-negative.')
        if self.alpha_max < 0.0:
            raise ValueError('dct_alpha_max must be non-negative.')
        if requested_active_bands < 0:
            raise ValueError('active_bands must be non-negative.')
        self.active_bands = min(requested_active_bands, self.num_bands)

    @staticmethod
    def _device_key(device: torch.device) -> str:
        index = '' if device.index is None else f':{device.index}'
        return f'{device.type}{index}'

    @classmethod
    def _dct_basis(cls, length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (int(length), cls._device_key(device), dtype)
        cached = cls._basis_cache.get(key)
        if cached is not None:
            return cached

        n = torch.arange(length, device=device, dtype=dtype)
        k = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
        basis = torch.cos(torch.pi * (n + 0.5) * k / float(length))
        basis[0, :] *= (1.0 / float(length)) ** 0.5
        if length > 1:
            basis[1:, :] *= (2.0 / float(length)) ** 0.5
        cls._basis_cache[key] = basis
        return basis

    @classmethod
    def _dct_basis_pair(
        cls,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        key = (int(height), int(width), cls._device_key(device), dtype)
        cached = cls._basis_pair_cache.get(key)
        if cached is not None:
            return cached
        basis_pair = (
            cls._dct_basis(height, device, dtype),
            cls._dct_basis(width, device, dtype),
        )
        cls._basis_pair_cache[key] = basis_pair
        return basis_pair

    @staticmethod
    def _smooth_step(x: torch.Tensor) -> torch.Tensor:
        x = torch.clamp(x, 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)

    @classmethod
    def _smooth_transition(cls, rho: torch.Tensor, edge: float, width: float) -> torch.Tensor:
        if width <= 0.0:
            return (rho >= edge).to(dtype=rho.dtype)
        return cls._smooth_step((rho - edge + width) / (2.0 * width))

    @classmethod
    def _band_masks(
        cls,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
        band_edges: Tuple[float, ...],
        smooth_width: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = (
            int(height),
            int(width),
            cls._device_key(device),
            dtype,
            tuple(float(item) for item in band_edges),
            float(smooth_width),
        )
        cached = cls._mask_cache.get(key)
        if cached is not None:
            return cached

        u = torch.arange(height, device=device, dtype=dtype) / float(max(height - 1, 1))
        v = torch.arange(width, device=device, dtype=dtype) / float(max(width - 1, 1))
        rho = torch.sqrt(u[:, None].pow(2) + v[None, :].pow(2))

        smooth_masks = []
        hard_masks = []
        for low, high in zip(band_edges[:-1], band_edges[1:]):
            left = cls._smooth_transition(rho, low, smooth_width)
            right = 1.0 - cls._smooth_transition(rho, high, smooth_width)
            smooth_masks.append(left * right)

            hard_left = cls._smooth_transition(rho, low, 0.0)
            hard_right = 1.0 - cls._smooth_transition(rho, high, 0.0)
            hard_masks.append(hard_left * hard_right)

        stacked_smooth_masks = torch.stack(smooth_masks, dim=0)
        stacked_hard_masks = torch.stack(hard_masks, dim=0)
        low_freeze_mask = rho < band_edges[0]
        cls._mask_cache[key] = (stacked_smooth_masks, stacked_hard_masks, low_freeze_mask)
        return stacked_smooth_masks, stacked_hard_masks, low_freeze_mask

    @classmethod
    def dct2(cls, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f'GlobalDCTBandGain expects BCHW tensors, got shape {tuple(x.shape)}.')
        height, width = x.shape[-2:]
        basis_h, basis_w = cls._dct_basis_pair(height, width, x.device, x.dtype)
        return torch.einsum('uh,bchw,vw->bcuv', basis_h, x, basis_w)

    @classmethod
    def idct2(cls, coeffs: torch.Tensor) -> torch.Tensor:
        if coeffs.ndim != 4:
            raise ValueError(f'GlobalDCTBandGain expects BCHW tensors, got shape {tuple(coeffs.shape)}.')
        height, width = coeffs.shape[-2:]
        basis_h, basis_w = cls._dct_basis_pair(height, width, coeffs.device, coeffs.dtype)
        return torch.einsum('uh,bcuv,vw->bchw', basis_h, coeffs, basis_w)

    def _normalize_params(self, params: torch.Tensor, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        params = params.to(device=device, dtype=dtype)
        if params.ndim == 1:
            params = params.unsqueeze(0)
        if params.ndim != 2 or params.shape[-1] != self.param_dim:
            raise ValueError(
                f'dct params must have shape [{self.param_dim}] or [B, {self.param_dim}], got {tuple(params.shape)}.'
            )
        if params.shape[0] == 1 and batch_size != 1:
            params = params.expand(batch_size, -1)
        elif params.shape[0] != batch_size:
            raise ValueError(f'dct params batch size {params.shape[0]} does not match input batch size {batch_size}.')
        return params

    def build_gain_map(self, x_shape: Tuple[int, int, int, int], params: torch.Tensor, device=None, dtype=None) -> torch.Tensor:
        if len(x_shape) != 4:
            raise ValueError(f'x_shape must be BCHW, got {x_shape}.')
        batch_size, _, height, width = x_shape
        if device is None:
            device = params.device
        if dtype is None:
            dtype = params.dtype
        device = torch.device(device)
        params = self._normalize_params(params, batch_size, device, dtype)

        smooth_masks, _, low_freeze_mask = self._band_masks(
            height,
            width,
            device,
            dtype,
            self.band_edges,
            self.smooth_width,
        )

        global_gain = 1.0 + params[:, 0].view(batch_size, 1, 1, 1)
        band_scale = self.alpha_max if self.alpha_max > 0.0 else 1.0
        band_delta = params[:, 1:1 + self.num_bands] * band_scale
        if self.active_bands < self.num_bands:
            limited_band_delta = torch.zeros_like(band_delta)
            limited_band_delta[:, :self.active_bands] = band_delta[:, :self.active_bands]
            band_delta = limited_band_delta

        # Use the raw input params almost linearly so the caller can directly steer
        # DCT coefficients, with only a broad stability clamp left in place.
        gain = global_gain
        if self.num_bands > 0:
            gain = gain + (
                band_delta.view(batch_size, self.num_bands, 1, 1) * smooth_masks.unsqueeze(0)
            ).sum(dim=1, keepdim=True)

        if self.preserve_low:
            gain = torch.where(low_freeze_mask.view(1, 1, height, width), torch.ones_like(gain), gain)
        gain = torch.nan_to_num(gain, nan=1.0, posinf=8.0, neginf=-8.0)
        return torch.clamp(gain, min=-8.0, max=8.0)

    def forward(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f'GlobalDCTBandGain expects BCHW tensors, got shape {tuple(x.shape)}.')
        params = self._normalize_params(params, x.shape[0], x.device, x.dtype)
        coeffs = self.dct2(x)
        gain = self.build_gain_map(tuple(x.shape), params, device=x.device, dtype=x.dtype)
        return self.idct2(coeffs * gain)
