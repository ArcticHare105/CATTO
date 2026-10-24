"""Optional arguments and input/output handling; SeeSR model knobs stay intact."""
import json
from pathlib import Path
import random

from .config import load_config


def add_arguments(parser):
    group = parser.add_argument_group("CATTO (enabled only when a config or windows are supplied)")
    group.add_argument("--catto-config", help="JSON window/method configuration")
    group.add_argument("--catto-windows", help="Override windows: start:length:candidates:scale:mode;...")
    group.add_argument("--catto-topk", type=int)
    group.add_argument("--catto-weights", nargs=3, type=float, metavar=("PSNR", "CONDITION", "TRAJECTORY"))
    group.add_argument("--catto-reuse", choices=["none", "selected", "decayed"])
    group.add_argument("--catto-reuse-decay", type=float, help="Reuse decay (default 0.1)")
    group.add_argument("--catto-save-conditions", action="store_true", default=None,
                       help="Save one retained condition PNG per step, including unchanged steps")


def validate_arguments(args):
    if not (args.catto_config or args.catto_windows):
        if any(getattr(args, name) is not None for name in
               ("catto_topk", "catto_weights", "catto_reuse", "catto_reuse_decay", "catto_save_conditions")):
            raise ValueError("supply --catto-config or --catto-windows to enable CATTO")
        args._catto_config = None
        return
    args._catto_config = load_config(
        args.catto_config, args.catto_windows, topk=args.catto_topk,
        consistency_weights=args.catto_weights, condition_reuse=args.catto_reuse,
        reuse_decay=args.catto_reuse_decay, save_conditions=args.catto_save_conditions,
    ).validate_schedule(args.num_inference_steps)


def attach(pipeline, args, device):
    config = getattr(args, "_catto_config", None)
    if config is None:
        return
    import pyiqa
    pipeline.catto_reward_model = pyiqa.create_metric("clipiqa", device=device).to(device).eval()
    pipeline.catto_config = config
    pipeline._catto_first_rng = None
    print(f"[CATTO] windows={len(config.windows)} NFE={config.nfe(args.num_inference_steps)} "
          f"reuse={config.condition_reuse} decay={config.reuse_decay}", flush=True)


def before_sample(pipeline, args, image_name, sample_index, generator):
    config = getattr(pipeline, "catto_config", None)
    if config is None:
        return
    import numpy as np
    import torch
    if config.independent_image_rng:
        if pipeline._catto_first_rng is None:
            pipeline._catto_first_rng = dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(),
                                            python=random.getstate(), numpy=np.random.get_state())
        saved = pipeline._catto_first_rng
        torch.set_rng_state(saved["cpu"])
        torch.cuda.set_rng_state(saved["cuda"])
        random.setstate(saved["python"])
        np.random.set_state(saved["numpy"])
    # This is CATTO's experimental per-image seed convention. The original
    # SeeSR entry point and generator behavior remain unchanged when disabled.
    generator.manual_seed((args.seed or 0) + sample_index)
    folder = Path(args.output_dir) / "catto" / Path(image_name).stem / f"sample{sample_index:02d}"
    folder.mkdir(parents=True, exist_ok=True)
    pipeline._catto_sample_dir = folder


def after_sample(pipeline, args, image_name, sample_index):
    if getattr(pipeline, "catto_config", None) is None:
        return
    folder = Path(pipeline._catto_sample_dir)
    result = dict(image=Path(image_name).name, sample=sample_index, seed=args.seed,
                  seesr=dict(num_inference_steps=args.num_inference_steps,
                             guidance_scale=args.guidance_scale, conditioning_scale=args.conditioning_scale,
                             start_point=args.start_point, start_steps=args.start_steps,
                             process_size=args.process_size, upscale=args.upscale,
                             align_method=args.align_method, mixed_precision=args.mixed_precision),
                  **pipeline.catto_diagnostics)
    (folder / "diagnostics.json").write_text(json.dumps(result, indent=2) + "\n")
