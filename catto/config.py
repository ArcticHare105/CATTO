"""Validated CATTO windows; all step indices are zero based."""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

MODES = ("consistency", "reward", "reward_only")
POLICIES = dict(zip(MODES, ("consistency_then_reward", "reward_then_consistency", "reward_only")))


def validate_decay(value):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("reuse decay must be finite and between 0 and 1")
    return value


def integer(value, name, minimum):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True)
class Window:
    start: int
    length: int
    candidates: int
    scale: float
    mode: str

    def __post_init__(self):
        integer(self.start, "window start", 0)
        integer(self.length, "window length", 1)
        integer(self.candidates, "candidates (including fixed and working)", 2)
        if isinstance(self.scale, bool) or not math.isfinite(self.scale) or self.scale < 0:
            raise ValueError("window scale must be finite and nonnegative")
        if self.mode not in MODES:
            raise ValueError(f"window mode must be one of {MODES}")

    @property
    def end(self):
        return self.start + self.length - 1

    @property
    def policy(self):
        return POLICIES[self.mode]


@dataclass(frozen=True)
class Config:
    windows: tuple
    consistency_weights: tuple = (8, 1, 1)
    topk: int = 3
    decode_chunk_size: int = 1
    mse_floor: float = 1e-10
    condition_reuse: str = "decayed"
    reuse_decay: float = 0.1
    independent_image_rng: bool = True
    save_conditions: bool = False

    def __post_init__(self):
        if not self.windows or any(not isinstance(w, Window) for w in self.windows):
            raise ValueError("CATTO needs at least one window")
        for a, b in zip(self.windows, self.windows[1:]):
            if b.start <= a.end:
                raise ValueError("windows must be sorted and cannot overlap; adjacent windows are allowed")
        integer(self.topk, "topk", 1)
        integer(self.decode_chunk_size, "decode_chunk_size", 1)
        weights = self.consistency_weights
        if len(weights) != 3 or any(not math.isfinite(v) or v < 0 for v in weights) or sum(weights) <= 0:
            raise ValueError("consistency_weights must be three finite nonnegative numbers with positive sum")
        if not math.isfinite(self.mse_floor) or self.mse_floor <= 0 or self.mse_floor > 1:
            raise ValueError("mse_floor must be finite and in (0,1]")
        if self.condition_reuse not in {"none", "selected", "decayed"}:
            raise ValueError("condition_reuse must be none, selected or decayed")
        validate_decay(self.reuse_decay)
        for field in ("independent_image_rng", "save_conditions"):
            if not isinstance(getattr(self, field), bool):
                raise ValueError(f"{field} must be a boolean")

    def validate_schedule(self, steps):
        integer(steps, "num_inference_steps", 1)
        if self.windows[-1].end >= steps:
            raise ValueError("a window extends past num_inference_steps")
        return self

    def nfe(self, steps):
        self.validate_schedule(steps)
        return steps + sum(w.length * (w.candidates - 1) for w in self.windows)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        value["windows"] = tuple(Window(**w) for w in value.get("windows", []))
        if "consistency_weights" in value:
            value["consistency_weights"] = tuple(value["consistency_weights"])
        return cls(**value)


def parse_windows(spec):
    """start:length:candidates:scale:mode;..."""
    windows = []
    for token in spec.split(";"):
        parts = token.strip().split(":")
        if len(parts) != 5:
            raise ValueError("use start:length:candidates:scale:mode for every window")
        start, length, candidates, scale, mode = parts
        windows.append(Window(int(start), int(length), int(candidates), float(scale), mode))
    return tuple(windows)


def load_config(path=None, windows=None, **overrides):
    data = json.loads(Path(path).read_text()) if path else {}
    if windows is not None:
        data["windows"] = [asdict(w) for w in parse_windows(windows)]
    data.update({key: value for key, value in overrides.items() if value is not None})
    return Config.from_dict(data)
