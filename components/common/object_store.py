import io
import json
import os
import tarfile
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List

import boto3
from botocore.client import BaseClient
from botocore.exceptions import ClientError


DEFAULT_S3_ENDPOINT_URL = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio.platform.svc.cluster.local:9000")
DEFAULT_AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-east-1")


def object_store_enabled() -> bool:
    return bool(os.getenv("OBJECT_STORE_ENDPOINT_URL") or os.getenv("MLFLOW_S3_ENDPOINT_URL"))


def object_store_client() -> BaseClient:
    endpoint_url = os.getenv("OBJECT_STORE_ENDPOINT_URL", DEFAULT_S3_ENDPOINT_URL)
    aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        region_name=os.getenv("AWS_DEFAULT_REGION", DEFAULT_AWS_REGION),
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
    )


def ensure_bucket(bucket: str) -> None:
    client = object_store_client()
    try:
        client.head_bucket(Bucket=bucket)
    except ClientError:
        client.create_bucket(Bucket=bucket)


def upload_json(bucket: str, key: str, payload: Dict[str, Any]) -> None:
    ensure_bucket(bucket)
    object_store_client().put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(payload, indent=2).encode("utf-8"),
        ContentType="application/json",
    )


def download_json(bucket: str, key: str) -> Dict[str, Any]:
    response = object_store_client().get_object(Bucket=bucket, Key=key)
    return json.loads(response["Body"].read().decode("utf-8"))


def upload_bytes(bucket: str, key: str, payload: bytes, content_type: str = "application/octet-stream") -> None:
    ensure_bucket(bucket)
    object_store_client().put_object(Bucket=bucket, Key=key, Body=payload, ContentType=content_type)


def upload_file(bucket: str, key: str, source_path: str | Path) -> None:
    ensure_bucket(bucket)
    object_store_client().upload_file(str(source_path), bucket, key)


def download_file(bucket: str, key: str, destination_path: str | Path) -> Path:
    destination = Path(destination_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    object_store_client().download_file(bucket, key, str(destination))
    return destination


def list_keys(bucket: str, prefix: str) -> List[str]:
    client = object_store_client()
    paginator = client.get_paginator("list_objects_v2")
    keys: List[str] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            keys.append(item["Key"])
    return sorted(keys)


def load_json_objects(bucket: str, prefix: str) -> List[Dict[str, Any]]:
    client = object_store_client()
    records: List[Dict[str, Any]] = []
    for key in list_keys(bucket, prefix):
        response = client.get_object(Bucket=bucket, Key=key)
        records.append(json.loads(response["Body"].read().decode("utf-8")))
    return records


def create_tar_gz(source_dir: str | Path, output_path: str | Path) -> Path:
    source = Path(source_dir)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as archive:
        archive.add(source, arcname=source.name)
    return output


def extract_tar_gz(archive_path: str | Path, destination_dir: str | Path) -> Path:
    destination = Path(destination_dir)
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        archive.extractall(path=destination)
        members = [Path(member.name).parts[0] for member in archive.getmembers() if member.name]
    top_level = members[0] if members else ""
    return destination / top_level if top_level else destination


def upload_directory_as_tarball(bucket: str, key: str, source_dir: str | Path, tmp_dir: str | Path) -> str:
    tmp_archive = create_tar_gz(source_dir, Path(tmp_dir) / f"{Path(key).name}")
    upload_file(bucket, key, tmp_archive)
    return key


def download_and_extract_tarball(bucket: str, key: str, cache_dir: str | Path) -> Path:
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    archive_path = cache_root / Path(key).name
    download_file(bucket, key, archive_path)
    return extract_tar_gz(archive_path, cache_root)
