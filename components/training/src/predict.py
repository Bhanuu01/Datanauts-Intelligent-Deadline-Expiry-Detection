import re, argparse
from dateutil import parser as dateutil_parser
from transformers import pipeline

CONFIDENCE_THRESHOLD = 0.7

CLF_LABEL2ID = {"none": 0, "expiration": 1, "effective": 2, "renewal": 3}
CLF_ID2LABEL = {v: k for k, v in CLF_LABEL2ID.items()}

NER_LABELS   = ["O","B-EXP_DATE","I-EXP_DATE","B-START_DATE","I-START_DATE","B-DURATION","I-DURATION"]

ALLOWED_ENTITIES = {
    "expiration": {"EXP_DATE"},
    "effective":  {"START_DATE"},
    "renewal":    {"DURATION", "START_DATE"},
}


def _resolve_date(text):
    """Parse a date string to ISO format. Returns None on failure."""
    try:
        return dateutil_parser.parse(text, dayfirst=False).strftime("%Y-%m-%d")
    except Exception:
        return None


def _extract_entities(ner_output, allowed_types):
    """
    Collapse sub-token BIO tags from HF NER pipeline into entity spans.
    Returns list of {entity_type, text}.
    """
    entities, current_type, current_tokens = [], None, []
    for tok in ner_output:
        label = tok["entity"]
        if label.startswith("B-"):
            if current_type and current_type in allowed_types:
                entities.append({"entity_type": current_type, "text": " ".join(current_tokens)})
            current_type   = label[2:]
            current_tokens = [tok["word"].lstrip("##")]
        elif label.startswith("I-") and current_type == label[2:]:
            word = tok["word"]
            if word.startswith("##"):
                current_tokens[-1] += word[2:]
            else:
                current_tokens.append(word)
        else:
            if current_type and current_type in allowed_types:
                entities.append({"entity_type": current_type, "text": " ".join(current_tokens)})
            current_type, current_tokens = None, []
    if current_type and current_type in allowed_types:
        entities.append({"entity_type": current_type, "text": " ".join(current_tokens)})
    return entities


def predict(
    sentences,
    clf_model_path,
    ner_model_path,
    contract_id="unknown",
    confidence_threshold=CONFIDENCE_THRESHOLD,
):
    """
    Run deadline detection on a list of pre-split sentences.

    Args:
        sentences            : list[str] — pre-split sentence strings from inference pipeline
        clf_model_path       : path or HF hub id of classifier model
        ner_model_path       : path or HF hub id of NER model
        contract_id          : identifier for the document
        confidence_threshold : float, events below this are flagged uncertain

    Returns:
        dict with keys: contract_id, has_deadline, uncertain, events
    """
    if not sentences:
        return {"contract_id": contract_id, "has_deadline": False, "uncertain": False, "events": []}

    clf_pipe = pipeline(
        "text-classification",
        model=clf_model_path,
        top_k=None,
        device=-1,
    )
    ner_pipe = pipeline(
        "token-classification",
        model=ner_model_path,
        aggregation_strategy=None,
        device=-1,
    )

    groups = {}  # event_type → [(sentence, confidence, all_scores)]
    for sent in sentences:
        results = clf_pipe(sent)[0]
        best    = max(results, key=lambda x: x["score"])
        label   = best["label"].lower()
        score   = best["score"]
        if label == "none":
            continue
        if label not in groups:
            groups[label] = []
        groups[label].append((sent, score, {r["label"].lower(): r["score"] for r in results}))

    if not groups:
        return {"contract_id": contract_id, "has_deadline": False, "uncertain": False, "events": []}

    events    = []
    any_uncertain = False

    for event_type, items in groups.items():
        allowed   = ALLOWED_ENTITIES.get(event_type, set())
        best_sent, best_conf, best_scores = max(items, key=lambda x: x[1])
        uncertain = best_conf < confidence_threshold
        if uncertain:
            any_uncertain = True

        deadline_date = None
        deadline_type = "computable"

        for sent, _, _ in items:
            ner_out  = ner_pipe(sent)
            entities = _extract_entities(ner_out, allowed)
            for ent in entities:
                if ent["entity_type"] == "EXP_DATE":
                    parsed = _resolve_date(ent["text"])
                    if parsed:
                        deadline_date = parsed
                        deadline_type = "explicit"
                        break
                elif ent["entity_type"] == "START_DATE" and deadline_date is None:
                    parsed = _resolve_date(ent["text"])
                    if parsed:
                        deadline_date = parsed
                        deadline_type = "explicit"
                elif ent["entity_type"] == "DURATION":
                    deadline_type = "computable"
            if deadline_date:
                break

        events.append({
            "event_type":      event_type,
            "deadline_date":   deadline_date,
            "deadline_type":   deadline_type,
            "confidence":      round(best_conf, 4),
            "uncertain":       uncertain,
            "source_sentence": best_sent,
            "class_scores":    {k: round(v, 4) for k, v in best_scores.items()},
        })

    return {
        "contract_id": contract_id,
        "has_deadline": len(events) > 0,
        "uncertain":   any_uncertain,
        "events":      events,
    }


def main():
    parser = argparse.ArgumentParser(description="Run deadline detection on pre-split sentences.")
    parser.add_argument("--clf_model",   required=True, help="Path or HF hub ID of classifier model")
    parser.add_argument("--ner_model",   required=True, help="Path or HF hub ID of NER model")
    parser.add_argument("--sentences",   nargs="+",     required=True, help="List of sentence strings")
    parser.add_argument("--contract_id", default="unknown")
    parser.add_argument("--threshold",   type=float, default=CONFIDENCE_THRESHOLD)
    args = parser.parse_args()

    import json
    result = predict(
        sentences=args.sentences,
        clf_model_path=args.clf_model,
        ner_model_path=args.ner_model,
        contract_id=args.contract_id,
        confidence_threshold=args.threshold,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
