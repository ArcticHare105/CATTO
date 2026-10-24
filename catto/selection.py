"""Three CATTO policies, matching the experimental rank-based selector."""
import torch


def average_rank(values):
    """Percentile ranks with average ties; a larger value ranks higher."""
    if values.ndim != 1 or not values.numel() or not bool(torch.isfinite(values).all()):
        raise ValueError("rank values must be a finite nonempty vector")
    if values.numel() == 1:
        return torch.full_like(values, 0.5)
    lower = (values[None, :] < values[:, None]).sum(1)
    tied = (values[None, :] == values[:, None]).sum(1)
    return (2 * lower + tied - 1).float() / (2 * (values.numel() - 1))


@torch.no_grad()
def select(images, rewards, conditions, trajectory, reference, mode, weights=(8, 1, 1), topk=3, mse_floor=1e-10):
    """Fixed candidate0 is the admission baseline and empty-pool fallback.

    CLIPIQA is the sole reward. Generated-reference PSNR, condition MSE and
    guided-noise trajectory MSE form the percentile-normalized consistency.
    A reference is an earlier decoded x0, never the ground truth.
    """
    count = images.shape[0]
    if images.ndim != 4 or images.shape[1] != 3 or count < 2 or conditions.shape != images.shape or rewards.numel() != count or trajectory.numel() != count:
        raise ValueError("one decoded RGB image, condition, reward and trajectory loss per candidate are required")
    if mode not in {"consistency", "reward", "reward_only"}:
        raise ValueError("unsupported window mode")
    if reference.shape != images[:1].shape:
        raise ValueError("pre-window x0 must match the decoded candidate image dimensions")
    if not bool(torch.isfinite(reference).all()):
        raise FloatingPointError("pre-window x0 must be finite")
    from .config import Config, Window
    # Keep direct Python use as strictly validated as JSON/CLI configuration.
    Config((Window(0, 1, count, 0, mode),), consistency_weights=weights,
           topk=topk, mse_floor=mse_floor)
    rewards, trajectory = rewards.reshape(-1).float(), trajectory.reshape(-1).float()
    condition_loss = (conditions.float() - conditions[:1].float()).square().flatten(1).mean(1)
    valid = (torch.isfinite(images).flatten(1).all(1) & torch.isfinite(rewards)
             & torch.isfinite(condition_loss) & torch.isfinite(trajectory))
    indices = valid.nonzero(as_tuple=False).flatten()
    if not indices.numel() or not bool(valid[0]):
        raise FloatingPointError("fixed candidate0 must have finite assessments")
    with torch.autocast(device_type=images.device.type, enabled=False):
        mse = (images[indices].float() - reference.float()).square().flatten(1).mean(1)
        psnr = -10 * torch.log10(mse.clamp_min(mse_floor))
    ranks = torch.stack((average_rank(psnr), average_rank(-condition_loss[indices]),
                         average_rank(-trajectory[indices])))
    denominator = 2 * (indices.numel() - 1) if indices.numel() > 1 else 2
    numerators = (ranks * denominator).round().double()
    weights = tuple(float(w) / sum(weights) for w in weights)
    consistency = (numerators * numerators.new_tensor(weights).view(3, 1)).sum(0) / denominator
    reward = rewards[indices]
    eligible = indices != 0
    if mode == "consistency":
        eligible &= consistency > consistency[0]
        first, final = consistency, reward
    else:
        eligible &= reward > reward[0]
        first, final = reward, consistency if mode == "reward" else reward
    qualified = eligible.nonzero(as_tuple=False).flatten()
    if mode == "reward_only":
        pool = qualified
    else:
        order = torch.argsort(first[qualified], descending=True, stable=True)
        pool = qualified[order[:min(topk, qualified.numel())]]
    if pool.numel():
        shortlist = torch.zeros_like(eligible)
        shortlist[pool] = True
        tied = shortlist & (final == final[pool].max())
        selected = int(indices[int(tied.nonzero(as_tuple=False)[0])])
    else:
        selected = 0

    def values(tensor):
        return {int(i): float(v) for i, v in zip(indices.cpu(), tensor.cpu())}

    return selected, {
        "selected_index": selected, "mode": mode,
        "eligible_indices": indices[eligible].tolist(), "pool_indices": indices[pool].tolist(),
        "candidate_zero_fallback": selected == 0,
        "psnr": values(psnr), "reward": values(reward),
        "condition_consistency_loss": values(condition_loss[indices]),
        "trajectory_consistency_loss": values(trajectory[indices]),
        "consistency_score": values(consistency),
        "consistency_components": {"reference": values(ranks[0]), "condition": values(ranks[1]), "trajectory": values(ranks[2])},
        "consistency_weights": list(weights), "topk": topk,
    }
