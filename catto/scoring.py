"""Use original SeeSR color alignment for CLIPIQA scoring only."""
import torch
from utils.wavelet_color_fix import adaptive_instance_normalization, wavelet_reconstruction

@torch.no_grad()
def output_aligned_images(images, fixed_condition, align_method="adain"):
    """Mirror PIL output quantization and color alignment using fixed LR.

    VAE image postprocessing rounds to 8-bit before color alignment. Existing
    ToPILImage color-fix output truncates to 8-bit afterward. Recover the
    original 8-bit style from the pipeline's possible half-precision LR copy.
    """
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("scoring images must be RGB NCHW")
    if (fixed_condition is None or fixed_condition.ndim != 4
            or fixed_condition.shape[0] != 1
            or fixed_condition.shape[1:] != images.shape[1:]):
        raise ValueError("output-aligned scoring requires one matching fixed LR condition")
    if align_method not in {"adain", "wavelet", "nofix"}:
        raise ValueError("unsupported output alignment method")
    with torch.autocast(device_type=images.device.type, enabled=False):
        content = images.float().clamp(0, 1).mul(255).round().div(255)
        style = fixed_condition.to(device=images.device, dtype=torch.float32)
        style = style.clamp(0, 1).mul(255).round().div(255)
        if align_method == "adain":
            result = adaptive_instance_normalization(content, style)
        elif align_method == "wavelet":
            result = wavelet_reconstruction(content, style)
        else:
            return content
        return result.clamp(0, 1).mul(255).floor().div(255)

