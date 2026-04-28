#!/usr/bin/env python3
import json
import os
import re
import base64
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
ARCHIVE_URL = os.getenv(
    "ONLINE_FEATURES_ARCHIVE_URL",
    "http://online-features.ml.svc.cluster.local:8000/archive-document",
)
INGEST_URL = os.getenv(
    "ONLINE_FEATURES_INGEST_URL",
    "http://online-features.ml.svc.cluster.local:8000/ingest",
)
RESULTS_DIR = Path(os.getenv("DEADLINE_RESULTS_DIR", "/usr/src/paperless/data/deadline-results"))
ASYNC_CHILD_ENV = "DEADLINE_POST_CONSUME_ASYNC_CHILD"
ASYNC_ENABLED = os.getenv("DEADLINE_POST_CONSUME_ASYNC", "true").lower() in {"1", "true", "yes"}
STATUS_REVIEW_TAG = "Status:Review Needed"
FEEDBACK_CORRECT_TAG = "Action:Accept"
FEEDBACK_WRONG_TAG = "Action:Reject"
EVENT_PRIORITY = {
    "deadline": 6,
    "expiration": 5,
    "renewal": 4,
    "effective": 3,
    "agreement": 2,
    "notice_period": 1,
    "unknown": 0,
}


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
    # Normalize OCR text to improve model date recognition (e.g., 3/28/12 -> March 28, 2012)
    normalized_content = normalize_ocr_text_for_model(content)

    payload = {
        "document_id": str(document_id),
        "ocr_text": normalized_content,
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


def post_ingest_event(document_id: int, content: str, document_type: str | None, filename: str | None) -> None:
    payload = {
        "document_id": str(document_id),
        "ocr_text": content,
        "document_type": document_type or "unknown",
        "filename": filename or f"document-{document_id}.pdf",
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        INGEST_URL,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=15):
        return


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


def archive_original_document(document_id: int, filename: str | None, original_path: str | None) -> None:
    if not original_path:
        return
    path = Path(original_path)
    if not path.exists() or not path.is_file():
        return
    payload = {
        "document_id": str(document_id),
        "filename": filename or path.name or f"document-{document_id}.pdf",
        "content_type": "application/pdf",
        "content_base64": base64.b64encode(path.read_bytes()).decode("ascii"),
    }
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        ARCHIVE_URL,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60):
        return


def build_tags(result: Dict[str, Any]) -> List[str]:
    tags = []
    if result.get("uncertain"):
        tags.append(STATUS_REVIEW_TAG)

    selected_events = dedupe_events_by_date(result.get("events", []))
    primary_event = select_primary_event(selected_events)
    if primary_event is not None:
        primary_label = str(primary_event.get("event_type", "unknown")).strip().lower() or "unknown"
        tags.append(f"Type:{primary_label.replace('_', ' ').title()}")

    for event in selected_events:
        tags.extend(build_event_tags(event))

    return sorted(set(tags))


def event_sort_key(event: Dict[str, Any], normalized_date: str) -> tuple[int, float, float, str]:
    event_type = str(event.get("event_type", "unknown")).strip().lower() or "unknown"
    class_scores = event.get("class_scores") or {}
    type_score = float(class_scores.get(event_type, 0.0))
    confidence = float(event.get("confidence", 0.0) or 0.0)
    priority = EVENT_PRIORITY.get(event_type, 0)
    return (priority, confidence, type_score, normalized_date)


def select_primary_event(events: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    if not events:
        return None
    return max(events, key=lambda event: event_sort_key(event, str(event.get("deadline_date") or "")))


def dedupe_events_by_date(events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    selected_by_date: Dict[str, Dict[str, Any]] = {}
    passthrough_events: List[Dict[str, Any]] = []

    for event in events:
        candidates = []
        deadline_date = str(event.get("deadline_date") or "").strip()
        if deadline_date:
            candidates.append(deadline_date)
        for candidate in event.get("date_candidates", []) or []:
            candidate = str(candidate or "").strip()
            if candidate:
                candidates.append(candidate)

        normalized_candidates: List[str] = []
        seen_candidates = set()
        for candidate in candidates:
            normalized = normalize_deadline_date(candidate)
            if normalized and normalized not in seen_candidates:
                normalized_candidates.append(normalized)
                seen_candidates.add(normalized)

        if not normalized_candidates:
            passthrough_events.append(event)
            continue

        for normalized in normalized_candidates:
            current = selected_by_date.get(normalized)
            if current is None or event_sort_key(event, normalized) > event_sort_key(current, normalized):
                chosen = dict(event)
                chosen["deadline_date"] = normalized
                chosen["date_candidates"] = [normalized]
                selected_by_date[normalized] = chosen

    ordered_selected = [selected_by_date[key] for key in sorted(selected_by_date)]
    return passthrough_events + ordered_selected


def normalize_ocr_text_for_model(text: str) -> str:
    """
    Normalize OCR text to improve model date recognition.
    Converts various date formats to month-name format (March 28, 2012)
    that the NER model was trained to recognize.
    
    Handles:
    - Slash-separated with year: 3/28/12, 03/28/2012
    - Slash-separated without year: 3/14, 5/27 (adds current year 2026)
    - Written dates with year: November 30, 2012, Nov 30 2012
    - Written dates without year: November 30th, December 10 (adds current year 2026)
    - Variations: Nov/Nov./November, with/without commas, with/without ordinal suffixes (st/nd/rd/th)
    """
    from datetime import datetime

    # Current year for dates without year specified
    current_year = 2026

    month_names = {
        1: "January", 2: "February", 3: "March", 4: "April",
        5: "May", 6: "June", 7: "July", 8: "August",
        9: "September", 10: "October", 11: "November", 12: "December"
    }

    # Reverse mapping for month name to number
    month_abbrev = {
        "jan": 1, "january": 1,
        "feb": 2, "february": 2,
        "mar": 3, "march": 3,
        "apr": 4, "april": 4,
        "may": 5,
        "jun": 6, "june": 6,
        "jul": 7, "july": 7,
        "aug": 8, "august": 8,
        "sep": 9, "sept": 9, "september": 9,
        "oct": 10, "october": 10,
        "nov": 11, "november": 11,
        "dec": 12, "december": 12
    }

    def convert_slash_date_with_year(match):
        """Convert m/d/yy or mm/dd/yyyy to Month d, yyyy"""
        parts = match.group(0).split('/')
        try:
            month, day = int(parts[0]), int(parts[1])
            year = int(parts[2])
            # Handle 2-digit years: 00-30 -> 2000-2030, 31-99 -> 1931-1999
            if year < 100:
                year = 2000 + year if year <= 30 else 1900 + year

            if 1 <= month <= 12 and 1 <= day <= 31:
                month_name = month_names[month]
                return f"{month_name} {day}, {year}"
        except (ValueError, KeyError):
            pass
        return match.group(0)

    def convert_slash_date_no_year(match):
        """Convert m/d or mm/dd to Month d, 2026 (current year)"""
        parts = match.group(0).split('/')
        try:
            month, day = int(parts[0]), int(parts[1])
            if 1 <= month <= 12 and 1 <= day <= 31:
                month_name = month_names[month]
                return f"{month_name} {day}, {current_year}"
        except (ValueError, KeyError):
            pass
        return match.group(0)

    def convert_written_date_with_year(match):
        """Convert written dates with year: November 30, 2012 or Nov 30 2012"""
        text_match = match.group(0)
        # Extract month name (with or without period)
        month_match = re.search(r'\b([a-z]+)\.?', text_match, re.IGNORECASE)
        if not month_match:
            return text_match
        
        month_str = month_match.group(1).lower()
        month_num = month_abbrev.get(month_str)
        if not month_num:
            return text_match
        
        # Extract day (with or without ordinal suffix)
        day_match = re.search(r'\b(\d{1,2})(?:st|nd|rd|th)?\b', text_match)
        if not day_match:
            return text_match
        
        day = int(day_match.group(1))
        if not (1 <= day <= 31):
            return text_match
        
        # Extract year
        year_match = re.search(r'\b(19\d{2}|20\d{2})\b', text_match)
        if not year_match:
            return text_match
        
        year = int(year_match.group(1))
        month_name = month_names[month_num]
        return f"{month_name} {day}, {year}"

    def convert_written_date_no_year(match):
        """Convert written dates without year: November 30th, December 10"""
        text_match = match.group(0)
        # Extract month name (with or without period)
        month_match = re.search(r'\b([a-z]+)\.?', text_match, re.IGNORECASE)
        if not month_match:
            return text_match
        
        month_str = month_match.group(1).lower()
        month_num = month_abbrev.get(month_str)
        if not month_num:
            return text_match
        
        # Extract day (with or without ordinal suffix)
        day_match = re.search(r'\b(\d{1,2})(?:st|nd|rd|th)?\b', text_match)
        if not day_match:
            return text_match
        
        day = int(day_match.group(1))
        if not (1 <= day <= 31):
            return text_match
        
        month_name = month_names[month_num]
        return f"{month_name} {day}, {current_year}"

    # Apply normalizations in order of specificity
    normalized = text
    
    # 1. Slash dates with year: m/d/yy or mm/dd/yyyy
    normalized = re.sub(r'\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b', convert_slash_date_with_year, normalized)
    
    # 2. Written dates with year: Month d, yyyy or Month d yyyy (handles Jan, January, etc.)
    normalized = re.sub(
        r'\b(jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|sept|september|oct|october|nov|november|dec|december)\.?\s+(\d{1,2})(?:st|nd|rd|th)?\s*,?\s*(19\d{2}|20\d{2})\b',
        convert_written_date_with_year,
        normalized,
        flags=re.IGNORECASE
    )
    
    # 3. Written dates without year: Month d, or Month dth (handles Jan, January, etc.)
    # Negative lookahead to avoid matching dates that already have years nearby
    normalized = re.sub(
        r'\b(jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|sept|september|oct|october|nov|november|dec|december)\.?\s+(\d{1,2})(?:st|nd|rd|th)?\b(?!\s*,?\s*(?:19|20)\d{2})',
        convert_written_date_no_year,
        normalized,
        flags=re.IGNORECASE
    )
    
    # 4. Slash dates without year: m/d or mm/dd (must come after year-based ones)
    normalized = re.sub(r'\b(\d{1,2})/(\d{1,2})(?!/)\b', convert_slash_date_no_year, normalized)
    
    return normalized


def normalize_deadline_date(raw_date: str) -> str:
    cleaned = raw_date.strip()
    if not cleaned:
        return ""
    cleaned = re.sub(r"\bsept\b", "sep", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("Sept.", "Sep.").replace("sept.", "sep.")
    lowered = cleaned.lower()
    if lowered in {"year", "month", "day"}:
        return ""
    if re.fullmatch(r"\d{1,3}", cleaned):
        return ""
    if re.fullmatch(r"\d{4}", cleaned):
        return ""
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", cleaned):
        return cleaned
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%b %d %Y"):
        try:
            from datetime import datetime

            return datetime.strptime(cleaned, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return re.sub(r"[^0-9A-Za-z_-]+", "-", cleaned).strip("-")


def build_event_tags(event: Dict[str, Any]) -> List[str]:
    event_type = str(event.get("event_type", "unknown")).strip().lower() or "unknown"
    event_label = event_type.replace("_", " ").title()
    tags = []

    candidates = []
    deadline_date = (event.get("deadline_date") or "").strip()
    if deadline_date:
        candidates.append(deadline_date)
    for candidate in event.get("date_candidates", []) or []:
        candidate = (candidate or "").strip()
        if candidate:
            candidates.append(candidate)

    seen = set()
    for candidate in candidates:
        normalized = normalize_deadline_date(candidate)
        if normalized and normalized not in seen:
            tags.append(f"{event_label}:{normalized}")
            seen.add(normalized)

    return tags


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
    import time

    document_id = resolve_document_id()
    try:
        if ASYNC_ENABLED and os.getenv(ASYNC_CHILD_ENV) != "1":
            spawn_async_worker(document_id)
            print(f"Queued asynchronous deadline inference for document {document_id}.")
            return 0

        # Retry fetching document content (handle race condition with Paperless OCR)
        content = ""
        max_retries = 5
        retry_delay = 1.0  # seconds

        for attempt in range(max_retries):
            document = get_document(document_id)
            content = (document.get("content") or "").strip()

            if content:
                if attempt > 0:
                    print(f"Document {document_id} content available after {attempt} retry/retries.")
                break

            if attempt < max_retries - 1:
                print(f"Document {document_id} content not yet available; retrying in {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                print(f"No OCR content available for document {document_id} after {max_retries} retries; skipping inference.")
                return 0

        if not content:
            print(f"No OCR content available for document {document_id}; skipping inference.")
            return 0

        filename = document.get("original_file_name") or document.get("archived_file_name")
        original_path = sys.argv[3] if len(sys.argv) > 3 else None
        try:
            post_ingest_event(
                document_id=document_id,
                content=content,
                document_type=document.get("document_type"),
                filename=filename,
            )
        except Exception as exc:
            print(
                f"Ingest capture failed for document {document_id}: {exc}",
                file=sys.stderr,
            )
        try:
            archive_original_document(
                document_id=document_id,
                filename=filename,
                original_path=original_path,
            )
        except Exception as exc:
            print(
                f"Original PDF archive failed for document {document_id}: {exc}",
                file=sys.stderr,
            )

        try:
            result = call_inference(
                document_id=document_id,
                content=content,
                document_type=document.get("document_type"),
                filename=filename,
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
