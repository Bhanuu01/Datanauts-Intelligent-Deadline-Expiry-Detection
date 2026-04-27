import argparse
import json
import os
from pathlib import Path

from components.common.object_store import upload_directory_as_tarball
from components.common.object_store import upload_json


DEFAULT_SOURCE_DIR = "/srv/datanauts-volumes/ml/model-storage/onnx_quantized_model"
DEFAULT_BUCKET = "datanauts-models"
DEFAULT_BUNDLE_KEY = "releases/bootstrap/bundle.tar.gz"
DEFAULT_RUNTIME_BUCKET = "datanauts-runtime"
DEFAULT_RELEASE_STATE_KEY = "automation/release_state.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish a local ONNX model directory as the bootstrap serving bundle."
    )
    parser.add_argument(
        "--source-dir",
        default=os.getenv("BOOTSTRAP_MODEL_SOURCE_DIR", DEFAULT_SOURCE_DIR),
        help="Local directory containing onnx_quantized_clf and onnx_quantized_ner",
    )
    parser.add_argument(
        "--bucket",
        default=os.getenv("MODEL_ARTIFACT_BUCKET", DEFAULT_BUCKET),
        help="Object-store bucket for model artifacts",
    )
    parser.add_argument(
        "--bundle-key",
        default=os.getenv("BOOTSTRAP_MODEL_BUNDLE_S3_KEY", DEFAULT_BUNDLE_KEY),
        help="Object-store key for the bootstrap bundle tarball",
    )
    parser.add_argument(
        "--runtime-bucket",
        default=os.getenv("RUNTIME_LOG_BUCKET", DEFAULT_RUNTIME_BUCKET),
        help="Object-store bucket for release state metadata",
    )
    parser.add_argument(
        "--release-state-key",
        default=os.getenv("RELEASE_STATE_S3_KEY", DEFAULT_RELEASE_STATE_KEY),
        help="Object-store key for release_state.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_dir = Path(args.source_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"Bootstrap model source directory not found: {source_dir}")
    if not (source_dir / "onnx_quantized_clf").exists():
        raise FileNotFoundError("Missing onnx_quantized_clf in bootstrap source directory")
    if not (source_dir / "onnx_quantized_ner").exists():
        raise FileNotFoundError("Missing onnx_quantized_ner in bootstrap source directory")

    bundle_key = upload_directory_as_tarball(
        bucket=args.bucket,
        key=args.bundle_key,
        source_dir=source_dir,
        tmp_dir="/tmp",
    )
    release_state = {
        "current_version": "bootstrap",
        "previous_version": None,
        "current_stage": os.getenv("CURRENT_RELEASE_STAGE", "staging"),
        "current_bundle_s3_key": bundle_key,
        "previous_bundle_s3_key": None,
        "last_action": "bootstrap_publish",
    }
    upload_json(args.runtime_bucket, args.release_state_key, release_state)
    print(json.dumps({"bundle_key": bundle_key, "release_state": release_state}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
