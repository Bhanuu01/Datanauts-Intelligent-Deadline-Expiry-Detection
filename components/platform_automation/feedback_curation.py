import json
import os
import re
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List
from typing import Tuple

from dateutil import parser as date_parser

from components.common.object_store import list_keys
from components.common.object_store import load_json_objects
from components.common.object_store import object_store_enabled
from components.common.object_store import object_store_client
from components.common.object_store import upload_json


DEFAULT_RUNTIME_LOG_BUCKET = "datanauts-runtime"
DEFAULT_ONLINE_FEEDBACK_PREFIX = "runtime/online-features/feedback"
DEFAULT_SERVING_FEEDBACK_PREFIX = "runtime/serving/feedback"
DEFAULT_INGEST_PREFIX = "runtime/online-features/ingest"
DATE_CANDIDATE_RE = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+\d{1,2},?\s+\d{4}"
    r"|\b(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+\d{4}"
    r"|\b\d{1,2}/\d{1,2}/\d{2,4}"
    r"|\b\d{4}-\d{2}-\d{2}",
    re.IGNORECASE,
)
POSITIVE_EVENTS = {"accept", "confirm", "manual_add", "edit"}
NEGATIVE_EVENTS = {"dismiss", "reject"}


def _load_local_jsonl(path_str: str) -> List[Tuple[str, Dict[str, Any]]]:
    path = Path(path_str)
    if not path.exists():
        return []
    records: List[Tuple[str, Dict[str, Any]]] = []
    for idx, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            records.append((f"local:{path.name}:{idx}", json.loads(line)))
        except json.JSONDecodeError:
            continue
    return records


def load_json_records(bucket: str, prefix: str) -> List[Tuple[str, Dict[str, Any]]]:
    client = object_store_client()
    records: List[Tuple[str, Dict[str, Any]]] = []
    for key in list_keys(bucket, prefix):
        response = client.get_object(Bucket=bucket, Key=key)
        records.append((key, json.loads(response["Body"].read().decode("utf-8"))))
    return records


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = date_parser.parse(value)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except Exception:
        return None


def _record_timestamp(key: str, payload: Dict[str, Any]) -> datetime | None:
    payload_ts = _parse_timestamp(str(payload.get("timestamp", "")).strip())
    if payload_ts is not None:
        return payload_ts
    match = re.search(r"(\d{14,20})", key)
    if not match:
        return None
    raw = match.group(1)
    try:
        return datetime.strptime(raw[:14], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _sort_records(records: Iterable[Tuple[str, Dict[str, Any]]]) -> List[Tuple[str, Dict[str, Any]]]:
    return sorted(
        records,
        key=lambda item: (
            _record_timestamp(item[0], item[1]) or datetime.min.replace(tzinfo=timezone.utc),
            item[0],
        ),
    )


def _load_prefix_records(prefix: str, local_path_env: str, default_local_path: str) -> List[Tuple[str, Dict[str, Any]]]:
    if object_store_enabled():
        try:
            return load_json_records(
                os.getenv("RUNTIME_LOG_BUCKET", DEFAULT_RUNTIME_LOG_BUCKET),
                prefix,
            )
        except Exception:
            pass
    return _load_local_jsonl(os.getenv(local_path_env, default_local_path))


def _is_newer_than_checkpoint(record_key: str, payload: Dict[str, Any], checkpoint: Dict[str, Any]) -> bool:
    checkpoint_ts = _parse_timestamp(str(checkpoint.get("last_trained_feedback_timestamp", "")).strip())
    if checkpoint_ts is None:
        return True
    record_ts = _record_timestamp(record_key, payload)
    if record_ts is None:
        return True
    return record_ts > checkpoint_ts


def _normalize_event_name(payload: Dict[str, Any]) -> str:
    return str(payload.get("event", "")).strip().lower()


def _normalize_event_type(payload: Dict[str, Any]) -> str:
    value = (
        payload.get("corrected_event_type")
        or payload.get("event_type")
        or payload.get("original_prediction")
        or "none"
    )
    return str(value).strip().lower()


def _normalize_sentence(payload: Dict[str, Any], ingest_by_doc: Dict[str, Dict[str, Any]]) -> str:
    sentence = str(payload.get("source_sentence") or payload.get("sentence") or "").strip()
    if sentence:
        return sentence
    document_id = str(payload.get("document_id", "")).strip()
    ingest_record = ingest_by_doc.get(document_id, {})
    features = ingest_record.get("features", [])
    event_type = _normalize_event_type(payload)
    if not isinstance(features, list):
        return ""
    prioritized = []
    for feature in features:
        feature_sentence = str(feature.get("sentence", "")).strip()
        if not feature_sentence:
            continue
        section_header = str(feature.get("document_metadata", {}).get("section_header", "")).lower()
        if event_type != "none" and event_type.split("_")[0] in section_header:
            return feature_sentence
        prioritized.append(feature_sentence)
    return prioritized[0] if prioritized else ""


def _iso_or_original(value: str | None) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        return date_parser.parse(raw).strftime("%Y-%m-%d")
    except Exception:
        return raw


def _candidate_supervision_values(payload: Dict[str, Any]) -> List[str]:
    values: List[str] = []
    for key in [
        "corrected_deadline_text",
        "corrected_deadline_date",
        "user_corrected_value",
        "deadline_date",
    ]:
        value = _iso_or_original(payload.get(key))
        if value and value not in values:
            values.append(value)
    return values


def _event_type_to_entity(event_type: str, target_value: str) -> str:
    value_lower = target_value.lower()
    duration_like = bool(re.search(r"\b(day|days|month|months|year|years)\b", value_lower))
    if event_type == "expiration":
        return "EXP_DATE"
    if event_type in {"effective", "agreement"}:
        return "START_DATE"
    if event_type == "notice_period":
        return "DURATION" if duration_like else "NOTICE_DATE"
    if event_type == "renewal":
        return "DURATION" if duration_like else "START_DATE"
    return "EXP_DATE"


def _find_supervision_span(sentence: str, target_value: str) -> Tuple[int, int] | None:
    lowered_sentence = sentence.lower()
    lowered_target = target_value.lower()
    direct_idx = lowered_sentence.find(lowered_target)
    if direct_idx >= 0:
        return direct_idx, direct_idx + len(lowered_target)

    target_iso = _iso_or_original(target_value)
    for match in DATE_CANDIDATE_RE.finditer(sentence):
        if _iso_or_original(match.group(0)) == target_iso:
            return match.start(), match.end()
    return None


def _tokenize_with_spans(sentence: str) -> List[Tuple[str, int, int]]:
    return [(match.group(0), match.start(), match.end()) for match in re.finditer(r"\S+", sentence)]


def _build_ner_record(payload: Dict[str, Any], sentence: str, event_type: str) -> Dict[str, Any] | None:
    target_values = _candidate_supervision_values(payload)
    if event_type == "none" or not sentence or not target_values:
        return None
    tokens_with_spans = _tokenize_with_spans(sentence)
    if not tokens_with_spans:
        return None

    for target_value in target_values:
        span = _find_supervision_span(sentence, target_value)
        if span is None:
            continue
        entity_type = _event_type_to_entity(event_type, target_value)
        tags: List[int] = []
        active = False
        started = False
        for _, start, end in tokens_with_spans:
            overlaps = not (end <= span[0] or start >= span[1])
            if overlaps and not started:
                tags.append({
                    "EXP_DATE": 1,
                    "START_DATE": 3,
                    "DURATION": 5,
                    "NOTICE_DATE": 7,
                }[entity_type])
                started = True
                active = True
            elif overlaps and active:
                tags.append({
                    "EXP_DATE": 2,
                    "START_DATE": 4,
                    "DURATION": 6,
                    "NOTICE_DATE": 8,
                }[entity_type])
            else:
                tags.append(0)
                if active and start >= span[1]:
                    active = False
        if any(tag != 0 for tag in tags):
            return {
                "sentence": sentence,
                "tokens": [token for token, _, _ in tokens_with_spans],
                "ner_tags": tags,
                "ground_truth_date": target_value if re.match(r"^\d{4}-\d{2}-\d{2}$", _iso_or_original(target_value)) else "",
            }
    return None


def compile_feedback_training_additions(output_dir: str, checkpoint: Dict[str, Any] | None = None) -> Dict[str, Any]:
    checkpoint = checkpoint or {}
    online_prefix = os.getenv("FEEDBACK_S3_PREFIX", DEFAULT_ONLINE_FEEDBACK_PREFIX)
    serving_prefix = os.getenv("SERVING_FEEDBACK_S3_PREFIX", DEFAULT_SERVING_FEEDBACK_PREFIX)
    ingest_prefix = os.getenv("PRODUCTION_DATA_S3_PREFIX", DEFAULT_INGEST_PREFIX)

    feedback_records = _sort_records(
        _load_prefix_records(online_prefix, "FEEDBACK_LOG_PATH", "/tmp/feedback_events.jsonl")
        + _load_prefix_records(serving_prefix, "SERVING_FEEDBACK_LOG_PATH", "/tmp/serving_feedback.jsonl")
    )
    ingest_records = _sort_records(
        _load_prefix_records(ingest_prefix, "PRODUCTION_DATA_LOG_PATH", "/tmp/production_ingest.jsonl")
    )
    ingest_by_doc = {
        str(payload.get("document_id", "")).strip(): payload
        for _, payload in ingest_records
        if payload.get("document_id")
    }

    new_records = [
        (key, payload)
        for key, payload in feedback_records
        if _is_newer_than_checkpoint(key, payload, checkpoint)
    ]

    classifier_records: List[Dict[str, Any]] = []
    ner_records: List[Dict[str, Any]] = []
    used_feedback_records = 0
    latest_new_feedback_timestamp: datetime | None = None
    latest_used_feedback_timestamp: datetime | None = None

    for key, payload in new_records:
        record_timestamp = _record_timestamp(key, payload)
        if record_timestamp is not None and (
            latest_new_feedback_timestamp is None or record_timestamp > latest_new_feedback_timestamp
        ):
            latest_new_feedback_timestamp = record_timestamp
        event_name = _normalize_event_name(payload)
        if event_name not in POSITIVE_EVENTS | NEGATIVE_EVENTS:
            continue

        sentence = _normalize_sentence(payload, ingest_by_doc)
        event_type = _normalize_event_type(payload)
        if not sentence:
            continue

        if event_name in POSITIVE_EVENTS and event_type != "none":
            classifier_label = event_type
        elif event_name in NEGATIVE_EVENTS:
            classifier_label = "none"
        else:
            continue

        classifier_records.append({
            "sentence": sentence,
            "classifier_label": classifier_label,
            "contract_id": str(payload.get("document_id", "feedback")),
            "ground_truth_date": _iso_or_original(payload.get("corrected_deadline_date") or payload.get("deadline_date")),
            "feedback_source": key,
        })
        used_feedback_records += 1
        if record_timestamp is not None and (
            latest_used_feedback_timestamp is None or record_timestamp > latest_used_feedback_timestamp
        ):
            latest_used_feedback_timestamp = record_timestamp

        if event_name in POSITIVE_EVENTS:
            ner_record = _build_ner_record(payload, sentence, event_type)
            if ner_record is not None:
                ner_record["contract_id"] = str(payload.get("document_id", "feedback"))
                ner_record["feedback_source"] = key
                ner_records.append(ner_record)

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    classifier_path = output_root / "classifier_feedback_additions.jsonl"
    ner_path = output_root / "ner_feedback_additions.jsonl"

    classifier_path.write_text(
        "".join(json.dumps(record) + "\n" for record in classifier_records),
        encoding="utf-8",
    )
    ner_path.write_text(
        "".join(json.dumps(record) + "\n" for record in ner_records),
        encoding="utf-8",
    )

    summary = {
        "feedback_records_total": len(feedback_records),
        "feedback_records_new": len(new_records),
        "feedback_records_used": used_feedback_records,
        "classifier_additions": len(classifier_records),
        "ner_additions": len(ner_records),
        "classifier_additions_path": str(classifier_path),
        "ner_additions_path": str(ner_path),
        "latest_new_feedback_timestamp": latest_new_feedback_timestamp.isoformat()
        if latest_new_feedback_timestamp is not None else "",
        "latest_used_feedback_timestamp": latest_used_feedback_timestamp.isoformat()
        if latest_used_feedback_timestamp is not None else "",
    }

    if object_store_enabled():
        upload_json(
            os.getenv("RUNTIME_LOG_BUCKET", DEFAULT_RUNTIME_LOG_BUCKET),
            os.getenv("FEEDBACK_CURATION_SUMMARY_S3_KEY", "automation/feedback_curation_summary.json"),
            summary,
        )

    return summary
