"""
Centralised environment / config loader.

Reads infrastructure endpoints from config/config.yaml (non-secret).
Reads AWS credentials from environment variables only (never from source).

Usage in every src/*.py:
    from env_config import setup_env, CFG
    setup_env()          # sets os.environ before any mlflow import
    MLFLOW_URI = CFG["mlflow"]["tracking_uri"]
"""
import os
import sys
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_CONFIG_PATH = os.path.join(_HERE, "..", "config", "config.yaml")

def _load_yaml():
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)

CFG = _load_yaml()


def setup_env():
    """Set environment variables required by MLflow and boto3/MinIO.

    - MLFLOW_TRACKING_URI and MLFLOW_S3_ENDPOINT_URL come from config.yaml.
    - AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY must be provided by the
      deployment environment (Docker, CI secrets, shell export).
      If they are absent a clear RuntimeError is raised so the problem is
      obvious rather than silently using wrong credentials.
    """
    mlflow_uri  = CFG["mlflow"]["tracking_uri"]
    s3_endpoint = CFG["mlflow"]["s3_endpoint"]

    os.environ.setdefault("MLFLOW_TRACKING_URI",   mlflow_uri)
    os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", s3_endpoint)
    os.environ.setdefault("GIT_PYTHON_REFRESH",    "quiet")

    missing = [k for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY")
               if not os.environ.get(k)]
    if missing:
        raise RuntimeError(
            f"[env_config] Missing required environment variable(s): {missing}\n"
            "Set them in your shell, Docker env, or CI secrets before running.\n"
            "See .env.example for the full list."
        )
