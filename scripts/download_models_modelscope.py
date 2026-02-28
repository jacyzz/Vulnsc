import argparse
import os
from modelscope.hub.snapshot_download import snapshot_download


def main():
    parser = argparse.ArgumentParser(description="Download baseline pretrained models from ModelScope")
    parser.add_argument("--model-root", default="/disk1/hs/model", help="Local directory to store models")
    parser.add_argument(
        "--models",
        nargs="*",
        default=[
            "microsoft/codebert-base",
            "microsoft/graphcodebert-base",
            "microsoft/unixcoder-base",
        ],
        help="Model IDs on ModelScope",
    )
    args = parser.parse_args()

    os.makedirs(args.model_root, exist_ok=True)

    for model_id in args.models:
        model_name = model_id.split("/")[-1]
        target_dir = os.path.join(args.model_root, model_name)
        os.makedirs(target_dir, exist_ok=True)
        print(f"[INFO] downloading {model_id} -> {target_dir}")
        try:
            snapshot_download(model_id=model_id, cache_dir=target_dir)
            print(f"[OK] {model_id}")
        except Exception as error:
            print(f"[FAIL] {model_id}: {error}")
            print("[HINT] If this model ID is unavailable on ModelScope, replace --models with the mirrored ID used in your org.")


if __name__ == "__main__":
    main()
