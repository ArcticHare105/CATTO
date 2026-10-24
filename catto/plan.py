"""Print/validate a window plan without loading torch or model checkpoints."""
import argparse
import json
from .config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--windows", help="start:length:candidates:scale:mode;...")
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()
    config = load_config(args.config, args.windows).validate_schedule(args.steps)
    print(json.dumps(dict(num_inference_steps=args.steps, candidate_nfe=config.nfe(args.steps),
                         **config.to_dict()), indent=2))


if __name__ == "__main__":
    main()
