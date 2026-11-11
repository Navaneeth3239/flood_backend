"""Download a compatible flood-arrival checkpoint from Hugging Face.

Usage: python scripts/download_arrival_model.py REPO_ID --revision COMMIT
Only config.json and safetensors are accepted; repository Python is never run.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from services.arrival_model import load_model, validate_config


def download(repo, revision, destination):
    from huggingface_hub import HfApi, hf_hub_download
    import shutil
    commit = HfApi().model_info(repo, revision=revision).sha
    config_file = hf_hub_download(repo, "config.json", revision=commit)
    config = validate_config(json.loads(Path(config_file).read_text()))
    # Validate task compatibility before downloading any weights.
    weights = hf_hub_download(repo, "model.safetensors", revision=commit)
    output = Path(destination) / repo.replace("/", "--") / commit
    output.mkdir(parents=True, exist_ok=True)
    config.update(source_repo=repo, source_revision=commit)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    shutil.copyfile(weights, output / "model.safetensors")
    load_model(str(output))
    return output.resolve()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--destination", default=str(Path(__file__).resolve().parents[1] / "cache" / "arrival-models"))
    args = parser.parse_args()
    try:
        output = download(args.repo, args.revision, args.destination)
        print(f"Validated checkpoint: {output}\nSet TWIN_ARRIVAL_MODEL_DIR={output}")
    except Exception as error:
        parser.exit(1, f"Arrival model not installed: {error}\n")
