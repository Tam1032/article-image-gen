"""Download the model weights listed in models.yaml.

Usage:
    python scripts/download_models.py [--models-dir models] [--include-optional]

Downloads resume if interrupted. Set HF_TOKEN for gated repositories.
"""

import argparse
from pathlib import Path

import yaml
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Download model weights for ComfyUI.")
    parser.add_argument("--config", type=Path, default=ROOT / "models.yaml")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--include-optional", action="store_true")
    args = parser.parse_args()

    models = yaml.safe_load(args.config.read_text(encoding="utf-8"))["models"]
    for model in models:
        if model.get("optional") and not args.include_optional:
            print(f"- skip (optional): {model['name']}")
            continue

        dest_dir = args.models_dir / model["type"]
        if (dest_dir / model["file"]).exists():
            print(f"- already present: {dest_dir / model['file']}")
            continue

        print(f"- downloading {model['name']} (~{model.get('size_gb', '?')} GB) -> {dest_dir}")
        dest_dir.mkdir(parents=True, exist_ok=True)
        hf_hub_download(repo_id=model["repo"], filename=model["file"], local_dir=dest_dir)
    print("Done.")


if __name__ == "__main__":
    main()
