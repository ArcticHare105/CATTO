"""Reuse the selected DCT parameter update direction with decaying momentum.

Uses CATTO's DCT rule: after selecting p*, v = decay*v +
(1-decay)*(p* - p_before). At each ordinary step: p = clip(p+v), then
v *= decay. Conditions are rendered from the immutable input LR.
"""
import torch

from .config import validate_decay as validate_condition_reuse_decay
from .proposals import (
    DCT11_DIMENSIONS, DCTLRCondition, clip_dct11_parameters,
    sample_dct11_parameters,
)


class DCTParameterReuseState:
    space = "dct_parameters"

    def __init__(self, fixed_condition, mode="decayed", decay=0.1):
        if mode not in {"decayed", "selected", "none"}:
            raise ValueError("condition_reuse must be decayed, selected or none")
        self.mode, self.decay = mode, validate_condition_reuse_decay(decay)
        self.transform = DCTLRCondition(fixed_condition)
        self.params = torch.zeros(1, DCT11_DIMENSIONS, device=fixed_condition.device)
        self.velocity = torch.zeros_like(self.params)
        self.condition = self.transform.condition
        self.age = 0
        self.event_before = self.candidate_parameters = None

    def render_working(self, params):
        return self.transform.render(params)

    @property
    def residual(self):
        # Existing pipeline records whether there is any direction left to carry.
        return self.velocity if bool((self.velocity != 0).any()) else None

    def begin_event(self, count, noise_scale, generator=None):
        if self.event_before is not None:
            raise RuntimeError("DCT event already active")
        offsets = sample_dct11_parameters(count - 2, noise_scale,
                                         self.params.device, generator)
        self.event_before = self.params.detach().clone()
        proposals = clip_dct11_parameters(self.params + offsets)
        self.candidate_parameters = torch.cat((torch.zeros_like(self.params),
                                               self.params, proposals), dim=0).detach()
        anchors = torch.cat((self.transform.condition, self.condition), dim=0)
        if proposals.shape[0] == 0:
            return anchors.detach()
        return torch.cat((anchors, self.render_working(proposals)), dim=0).detach()

    def commit_selection(self, selected):
        if self.event_before is None:
            raise RuntimeError("No DCT event to commit")
        if not 0 <= selected < self.candidate_parameters.shape[0]:
            raise ValueError("DCT selection index out of bounds")
        chosen = self.candidate_parameters[selected:selected + 1]
        if self.mode == "none":
            self.params = self.event_before
            self.velocity.zero_()
        else:
            previous = self.event_before
            self.params = chosen.detach().clone()
            if self.mode == "decayed" and selected != 0:
                self.velocity = (self.decay * self.velocity
                                 + (1 - self.decay) * (self.params - previous)).detach()
            else:
                # Fixed LR fallback clears the prior direction too.
                self.velocity.zero_()
        self.event_before = None
        self.age = 0
        self.condition = self.render_working(self.params)
        return self.condition

    def advance(self):
        if self.event_before is not None:
            raise RuntimeError("DCT direction must not advance inside a key window")
        if self.mode != "decayed" or self.residual is None:
            return None
        self.params = clip_dct11_parameters(self.params + self.velocity).detach()
        self.velocity = (self.velocity * self.decay).detach()
        self.age += 1
        self.condition = self.render_working(self.params)
        return self.condition

    def diagnostics(self):
        record = {"condition_reuse_space": self.space,
                "dct_parameters": self.params[0].tolist(),
                "dct_update_direction": self.velocity[0].tolist(),
                "dct_direction_age": self.age,
                "dct_candidate_parameters": (self.candidate_parameters.tolist()
                    if self.candidate_parameters is not None else []),
                "dct_reuse_rule": "p=clip(p+v);v=decay*v",
                "dct_direction_selection_rule": "v=decay*v+(1-decay)*(p_selected-p_before)"}
        return record
