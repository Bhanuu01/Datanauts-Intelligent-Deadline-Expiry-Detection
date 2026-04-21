#!/usr/bin/env python3
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List


PAPERLESS_URL = os.getenv("PAPERLESS_INTERNAL_URL", "http://127.0.0.1:8000")
INFERENCE_URL = os.getenv(
    "DEADLINE_INFERENCE_URL",
    "http://deadline-onnx-serving.ml.svc.cluster.local:8004/predict",
)
FEEDBACK_URL = os.getenv(
    "ONLINE_FEATURES_FEEDBACK_URL",
    "http://online-features.ml.svc.cluster.local:8000/feedback",
)
RESULTS_DIR = Path(os.getenv("DEADLINE_RESULTS_DIR", "/usr/src/paperless/data/deadline-results"))
ASYNC_CHILD_ENV = "DEADLINE_POST_CONSUME_ASYNC_CHILD"
ASYNC_ENABLED = os.getenv("DEADLINE_POST_CONSUME_ASYNC", "true").lower() in {"1", "true", "yes"}
DEADLINE_TAG_PREFIX = "ML Deadline Date: "
REVIEW_PENDING_TAG = "ML Review Pending"
FEEDBACK_CORRECT_TAG = "ML Feedback Correct"
FEEDBACK_WRONG_TAG = "ML Feedback Wrong"
DEADLINE_DETECTED_TAG = "ML Deadline Detected"
REVIEW_NEEDED_TAG = "ML Review Needed"


def basic_auth_header() -> str:
    import base64

    username = os.getenv("PAPERLESS_ADMIN_USER", "admin")
    password = os.environ["PAPERLESS_ADMIN_PASSWORD"]
    token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
    return f"Basic {token}"


def request_json(method: str, url: str, payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    data = None
    headers = {
        "Authorization": basic_auth_header(),
        "Accept": "application/json",
    }
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def get_document(document_id: int) -> Dict[str, Any]:
    return request_json("GET", f"{PAPERLESS_URL}/api/documents/{document_id}/")


def find_tag_id(name: str) -> int | None:
    query = urllib.parse.urlencode({"name__iexact": name, "page_size": 1})
    response = request_json("GET", f"{PAPERLESS_URL}/api/tags/?{query}")
    results = response.get("results", [])
    return results[0]["id"] if results else None


def ensure_tag(name: str) -> int:
    existing = find_tag_id(name)
    if existing is not None:
        return existing

    created = request_json("POST", f"{PAPERLESS_URL}/api/tags/", {"name": name})
    return int(created["id"])


def patch_document_tags(document_id: int, existing_tags: List[int], extra_tag_ids: List[int]) -> None:
    merged_tags = sorted(set(existing_tags + extra_tag_ids))
    request_json("PATCH", f"{PAPERLESS_URL}/api/documents/{document_id}/", {"tags": merged_tags})


def call_inference(document_id: int, content: str, document_type: str | None, filename: str | None) -> Dict[str, Any]:
    payload = {
        "document_id": str(document_id),
        "ocr_text": content,
        "document_type": document_type or "unknown",
        "filename": filename or f"document-{document_id}.pdf",
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        INFERENCE_URL,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=float(os.getenv("INFERENCE_TIMEOUT_SECONDS", "20"))) as response:
        return json.loads(response.read().decode("utf-8"))


def post_feedback(payload: Dict[str, Any]) -> None:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        FEEDBACK_URL,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=15):
        return


def build_tags(result: Dict[str, Any]) -> List[str]:
    tags = []
    if result.get("has_deadline"):
        tags.append(DEADLINE_DETECTED_TAG)
        tags.append(REVIEW_PENDING_TAG)
    if result.get("uncertain"):
        tags.append(REVIEW_NEEDED_TAG)
        tags.append(REVIEW_PENDING_TAG)

    for event in result.get("events", []):
        event_type = event.get("event_type", "unknown")
        tags.append(f"ML Event: {str(event_type).replace('_', ' ').title()}")

    deadline_tag = build_deadline_date_tag(result)
    if deadline_tag:
        tags.append(deadline_tag)

    return sorted(set(tags))


def normalize_deadline_date(raw_date: str) -> str:
    cleaned = raw_date.strip()
    if not cleaned:
        return ""
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", cleaned):
        return cleaned
    for fmt in ("%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%b %d %Y"):
        try:
            from datetime import datetime

            return datetime.strptime(cleaned, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return re.sub(r"[^0-9A-Za-z_-]+", "-", cleaned).strip("-")


def build_deadline_date_tag(result: Dict[str, Any]) -> str | None:
    for event in result.get("events", []):
        candidates = []
        deadline_date = (event.get("deadline_date") or "").strip()
        if deadline_date:
            candidates.append(deadline_date)
        for candidate in event.get("date_candidates", []) or []:
            candidate = (candidate or "").strip()
            if candidate:
                candidates.append(candidate)

        for candidate in candidates:
            normalized = normalize_deadline_date(candidate)
            if normalized:
                return f"{DEADLINE_TAG_PREFIX}{normalized}"
    return None


def persist_result(document_id: int, result: Dict[str, Any]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"{document_id}.json"
    output_path.write_text(json.dumps(result, indent=2))


def record_feedback_events(document_id: int, result: Dict[str, Any]) -> None:
    event_name = "review_required" if result.get("uncertain") else (
        "auto_tagged" if result.get("has_deadline") else "no_deadline"
    )
    confidence = max((event.get("confidence", 0.0) for event in result.get("events", [])), default=0.0)
    event_types = sorted({event.get("event_type", "none") for event in result.get("events", [])}) or ["none"]
    for event_type in event_types:
        try:
            post_feedback(
                {
                    "event": event_name,
                    "document_id": str(document_id),
                    "event_type": event_type,
                    "confidence": confidence,
                    "notes": f"paperless_hook:{result.get('mode', 'unknown')}",
                }
            )
        except Exception as exc:
            print(
                f"Feedback capture failed for document {document_id} ({event_type}): {exc}",
                file=sys.stderr,
            )


def spawn_async_worker(document_id: int) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = RESULTS_DIR / f"{document_id}.hook.log"
    env = os.environ.copy()
    env[ASYNC_CHILD_ENV] = "1"
    env["DOCUMENT_ID"] = str(document_id)
    with log_path.open("a", encoding="utf-8") as handle:
        subprocess.Popen(
            [sys.executable, __file__],
            env=env,
            stdout=handle,
            stderr=handle,
            start_new_session=True,
            close_fds=True,
        )


def resolve_document_id() -> int:
    if os.getenv("DOCUMENT_ID"):
        return int(os.environ["DOCUMENT_ID"])
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        return int(sys.argv[1])
    raise KeyError("DOCUMENT_ID")


def main() -> int:
    document_id = resolve_document_id()
    try:
        if ASYNC_ENABLED and os.getenv(ASYNC_CHILD_ENV) != "1":
            spawn_async_worker(document_id)
            print(f"Queued asynchronous deadline inference for document {document_id}.")
            return 0

        document = get_document(document_id)
        content = (document.get("content") or "").strip()
        if not content:
            print(f"No OCR content available for document {document_id}; skipping inference.")
            return 0

        try:
            result = call_inference(
                document_id=document_id,
                content=content,
                document_type=document.get("document_type"),
                filename=document.get("original_file_name") or document.get("archived_file_name"),
            )
            persist_result(document_id, result)
            record_feedback_events(document_id, result)
        except Exception as exc:
            print(
                f"Skipping deadline inference for document {document_id} after non-fatal error: {exc}",
                file=sys.stderr,
            )
            return 0

        tag_names = build_tags(result)
        ensure_tag(FEEDBACK_CORRECT_TAG)
        ensure_tag(FEEDBACK_WRONG_TAG)
        if tag_names:
            tag_ids = [ensure_tag(name) for name in tag_names]
            patch_document_tags(document_id, document.get("tags", []), tag_ids)
            print(f"Updated document {document_id} with tags: {', '.join(tag_names)}")
        else:
            print(f"No deadline tags added for document {document_id}.")

        return 0
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"HTTP error during post-consume hook: {exc.code} {body}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Unexpected error during post-consume hook: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
