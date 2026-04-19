import os, re, json, random, argparse
from collections import Counter
from datasets import load_dataset, Dataset, DatasetDict

os.environ["GIT_PYTHON_REFRESH"] = "quiet"

DATASET_ID = "tanvitakavane/datanauts_project_cuad-deadline-ner-version2"
SAVE_PATH  = "./data/deadline_sentences"

NER_LABELS = ["O","B-EXP_DATE","I-EXP_DATE","B-START_DATE","I-START_DATE","B-DURATION","I-DURATION"]
NER_L2I    = {l: i for i, l in enumerate(NER_LABELS)}

CLF_L2I    = {"none": 0, "expiration": 1, "effective": 2, "renewal": 3}
CLF_I2L    = {0: "none", 1: "expiration", 2: "effective", 3: "renewal"}

MONTH_WORDS = {
    "january","february","march","april","may","june","july",
    "august","september","october","november","december",
    "jan","feb","mar","apr","jun","jul","aug","sep","oct","nov","dec",
}
DURATION_WORDS = {
    "year","years","month","months","day","days","week","weeks",
    "quarter","quarters","one","two","three","four","five","six",
    "seven","eight","nine","ten","twelve","thirty","sixty","ninety",
    "annual","annually",
}


def clean_clause(raw):
    if not raw or str(raw) in ("[]", "null", "None"):
        return ""
    try:
        parsed = json.loads(str(raw).replace("'", '"'))
        return " ".join(str(p) for p in parsed).strip() if isinstance(parsed, list) else str(parsed).strip()
    except Exception:
        return re.sub(r"^\[?['\"]?|['\"]?\]?$", "", str(raw).strip()).strip()


def sent_split(text):
    text = re.sub(r"\n{2,}", " ", text)
    text = re.sub(r"\n", " ", text)
    return [s.strip() for s in re.split(r"(?<=[.?!])\s+(?=[A-Z0-9\(])", text) if len(s.strip()) > 20]


def overlaps(sent, clause, threshold=0.5):
    if not clause:
        return False
    cw = set(clause.lower().split())
    sw = set(sent.lower().split())
    return len(cw & sw) / max(len(cw), 1) >= threshold


def bio_tag(tokens, ctype):
    tags = ["O"] * len(tokens)
    if ctype == "none":
        return tags
    if ctype == "expiration":
        b, i, check = "B-EXP_DATE",   "I-EXP_DATE",   lambda t: t.lower() in MONTH_WORDS or bool(re.fullmatch(r"\d{4}", t.rstrip(".,")))
    elif ctype == "effective":
        b, i, check = "B-START_DATE", "I-START_DATE",  lambda t: t.lower() in MONTH_WORDS or bool(re.fullmatch(r"\d{4}", t.rstrip(".,")))
    else:
        b, i, check = "B-DURATION",   "I-DURATION",    lambda t: t.lower().rstrip(".,;:") in DURATION_WORDS
    in_span = False
    for idx, tok in enumerate(tokens):
        if check(tok):
            tags[idx] = i if in_span else b
            in_span   = True
        else:
            in_span   = False
    return tags


def process_contract(row, split_name):
    exp_clause  = clean_clause(row.get("Expiration Date",  "") or "")
    exp_iso     = row.get("expiration_date_iso") or None
    eff_clause  = clean_clause(row.get("Effective Date",   "") or "")
    eff_iso     = row.get("effective_date_iso")  or None
    ren_clause  = clean_clause(row.get("Renewal Term",     "") or "")
    ocr         = row.get("ocr_text", "") or ""
    contract_id = row.get("Filename", "")
    event_type  = row.get("event_type", "") or ""
    examples    = []

    for sent in sent_split(ocr):
        tokens = sent.split()
        if len(tokens) < 5:
            continue
        if exp_clause and overlaps(sent, exp_clause):
            clf   = 1                                       # always "expiration" event type
            ctype = "expiration" if exp_iso else "duration"  # NER tags EXP_DATE if explicit, DURATION if computable
            gt    = exp_iso or ""
        elif eff_clause and overlaps(sent, eff_clause):
            clf, ctype, gt = 2, "effective", (eff_iso or "")
        elif ren_clause and overlaps(sent, ren_clause):
            clf, ctype, gt = 3, "duration", ""              # renewal → NER tags DURATION
        else:
            clf, ctype, gt = 0, "none", ""

        ner_tags = [NER_L2I[t] for t in bio_tag(tokens, ctype)]
        examples.append({
            "contract_id": contract_id, "split": split_name,
            "sentence": sent, "tokens": tokens,
            "ner_tags": ner_tags, "classifier_label": clf,
            "ground_truth_date": gt, "event_type": event_type,
        })
    return examples


def build_split(hf_split, name):
    rows = []
    for row in hf_split:
        rows.extend(process_contract(row, name))
    return Dataset.from_list(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", default=SAVE_PATH)
    args = parser.parse_args()

    print(f"Loading {DATASET_ID} ...")
    raw = load_dataset(DATASET_ID)
    print("Contracts — train:", len(raw["train"]), "| val:", len(raw["val"]), "| test:", len(raw["test"]))

    train_ds = build_split(raw["train"], "train")
    val_ds   = build_split(raw["val"],   "val")
    test_ds  = build_split(raw["test"],  "test")

    dd = DatasetDict({"train": train_ds, "val": val_ds, "test": test_ds})

    print("\n=== Sentence-level dataset ===")
    for name, ds in dd.items():
        c = Counter(ds["classifier_label"])
        print(f"  [{name}] {len(ds):>6} sentences | "
              f"none={c[0]:>5}  expiration={c[1]:>4}  effective={c[2]:>4}  renewal={c[3]:>4}")

    os.makedirs(args.save_path, exist_ok=True)
    dd.save_to_disk(args.save_path)

    os.makedirs("./data", exist_ok=True)
    with open("./data/label_maps.json", "w") as f:
        json.dump({
            "ner_label2id": NER_L2I,
            "ner_id2label": {str(k): v for k, v in enumerate(NER_LABELS)},
            "clf_label2id": CLF_L2I,
            "clf_id2label": {str(k): v for k, v in CLF_I2L.items()},
            "save_path": args.save_path,
        }, f, indent=2)

    print(f"\nSaved to {args.save_path}")
    print("Label maps saved to ./data/label_maps.json")


if __name__ == "__main__":
    main()
