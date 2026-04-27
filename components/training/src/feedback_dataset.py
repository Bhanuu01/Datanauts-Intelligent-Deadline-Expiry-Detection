import json
import os
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Iterable
from typing import List

from datasets import Dataset
from datasets import concatenate_datasets


CLASSIFIER_LABELS = {
    "none": 0,
    "expiration": 1,
    "effective": 2,
    "renewal": 3,
    "agreement": 4,
    "notice_period": 5,
}

NER_LABELS = {
    "O": 0,
    "B-EXP_DATE": 1,
    "I-EXP_DATE": 2,
    "B-START_DATE": 3,
    "I-START_DATE": 4,
    "B-DURATION": 5,
    "I-DURATION": 6,
    "B-NOTICE_DATE": 7,
    "I-NOTICE_DATE": 8,
}


def _load_jsonl(path_str: str | None) -> List[Dict[str, Any]]:
    if not path_str:
        return []
    path = Path(path_str)
    if not path.exists():
        return []
    records: List[Dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def _normalize_classifier_label(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value if value in CLASSIFIER_LABELS.values() else None
    normalized = str(value).strip().lower()
    return CLASSIFIER_LABELS.get(normalized)


def _normalize_tokens(record: Dict[str, Any]) -> List[str]:
    tokens = record.get("tokens")
    if isinstance(tokens, list) and tokens:
        return [str(token) for token in tokens]
    sentence = str(record.get("sentence", "")).strip()
    return sentence.split() if sentence else []


def _normalize_ner_tags(record: Dict[str, Any], token_count: int) -> List[int] | None:
    tags = record.get("ner_tags")
    if not isinstance(tags, list):
        return None
    normalized: List[int] = []
    for tag in tags:
        if isinstance(tag, int):
            normalized.append(tag)
            continue
        tag_id = NER_LABELS.get(str(tag).strip())
        if tag_id is None:
            return None
        normalized.append(tag_id)
    if len(normalized) != token_count:
        return None
    return normalized


def merge_classifier_feedback_additions(base_dataset, env_var: str = "FEEDBACK_CLASSIFIER_ADDITIONS_PATH"):
    additions = _load_jsonl(os.getenv(env_var))
    normalized: List[Dict[str, Any]] = []
    for record in additions:
        sentence = str(record.get("sentence", "")).strip()
        label = _normalize_classifier_label(record.get("classifier_label"))
        if not sentence or label is None:
            continue
        normalized.append({
            "sentence": sentence,
            "classifier_label": label,
            "contract_id": str(record.get("contract_id", "feedback")),
            "ground_truth_date": str(record.get("ground_truth_date", "")),
            "feedback_source": str(record.get("feedback_source", "feedback")),
        })
    if not normalized:
        return base_dataset
    additions_dataset = Dataset.from_list(normalized)
    keep_columns = [column for column in base_dataset.column_names if column in additions_dataset.column_names]
    additions_dataset = additions_dataset.select_columns(keep_columns)
    return concatenate_datasets([base_dataset, additions_dataset])


def merge_ner_feedback_additions(base_dataset, env_var: str = "FEEDBACK_NER_ADDITIONS_PATH"):
    additions = _load_jsonl(os.getenv(env_var))
    normalized: List[Dict[str, Any]] = []
    for record in additions:
        tokens = _normalize_tokens(record)
        if not tokens:
            continue
        ner_tags = _normalize_ner_tags(record, len(tokens))
        if ner_tags is None:
            continue
        normalized.append({
            "tokens": tokens,
            "ner_tags": ner_tags,
            "contract_id": str(record.get("contract_id", "feedback")),
            "sentence": str(record.get("sentence", " ".join(tokens))),
            "ground_truth_date": str(record.get("ground_truth_date", "")),
            "feedback_source": str(record.get("feedback_source", "feedback")),
        })
    if not normalized:
        return base_dataset
    additions_dataset = Dataset.from_list(normalized)
    keep_columns = [column for column in base_dataset.column_names if column in additions_dataset.column_names]
    additions_dataset = additions_dataset.select_columns(keep_columns)
    return concatenate_datasets([base_dataset, additions_dataset])
