import os, re, json, argparse, unicodedata
from collections import Counter
from datetime import datetime
from dateutil import parser as dateutil_parser
from datasets import load_dataset, Dataset, DatasetDict

os.environ["GIT_PYTHON_REFRESH"] = "quiet"

DATASET_ID = "tanvitakavane/datanauts_project_cuad-deadline-ner-version2"
SAVE_PATH  = "./data/deadline_sentences"

NER_LABELS = ["O","B-EXP_DATE","I-EXP_DATE","B-START_DATE","I-START_DATE","B-DURATION","I-DURATION"]
NER_L2I    = {l: i for i, l in enumerate(NER_LABELS)}

CLF_L2I = {"none": 0, "expiration": 1, "effective": 2, "renewal": 3}
CLF_I2L = {0: "none", 1: "expiration", 2: "effective", 3: "renewal"}

# ── Regex: Date patterns (8 formats) ───────────────────────────────────────
_ML  = r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
_MS  = r"(?:Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
_M   = r"(?:" + _ML + r"|" + _MS + r")\.?"
_ORD = r"(?:\d{1,2}(?:st|nd|rd|th))"

DATE_RE = re.compile("|".join([
    _M + r"\s+\d{1,2}[,.]?\s+\d{4}",               # January 1, 2024 / Jan. 1 2024
    r"\d{1,2}\s+" + _M + r"\s+\d{4}",               # 1 January 2024
    _ORD + r"\s+(?:of\s+)?" + _M + r"(?:\s+\d{4})?",# 31st December 2024
    r"\d{1,2}/\d{1,2}/\d{2,4}",                     # 12/31/2024  or  12/31/24
    r"\d{1,2}-\d{1,2}-\d{2,4}",                     # 12-31-2024
    r"\d{4}-\d{2}-\d{2}",                            # 2024-12-31 (ISO)
    r"\d{1,2}\.\d{1,2}\.\d{2,4}",                   # 31.12.2024
    _M + r"\s+\d{4}",                                # January 2024 (no day)
    r"\b(?:19|20)\d{2}\b",                           # 2024 — year-only fallback
]), re.IGNORECASE)

# ── Regex: Duration patterns ────────────────────────────────────────────────
_NUM_WORDS = r"(?:one|two|three|four|five|six|seven|eight|nine|ten|twelve|thirty|sixty|ninety)"
_UNIT      = r"(?:year|month|day|week|quarter)"

DURATION_RE = re.compile("|".join([
    r"\d+\s*\(\d+\)\s*" + _UNIT + r"s?",            # 3 (three) years / three (3) years
    _NUM_WORDS + r"\s*\(\d+\)\s*" + _UNIT + r"s?",
    r"\d+[\s\-]+" + _UNIT + r"s?",                  # 3 years / 3-year
    _NUM_WORDS + r"[\s\-]+" + _UNIT + r"s?",        # three years / one-year
    r"\d+[\s\-]*" + _UNIT + r"s?\s+(?:period|term)",# 12-month period
    _NUM_WORDS + r"[\s\-]*" + _UNIT + r"s?\s+(?:period|term)",
]), re.IGNORECASE)


# ── Text utilities ──────────────────────────────────────────────────────────

def normalise(text):
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_clause(raw):
    if not raw or str(raw) in ("[]", "null", "None"):
        return ""
    try:
        parsed = json.loads(str(raw).replace("'", '"'))
        return " ".join(str(p) for p in parsed).strip() if isinstance(parsed, list) else str(parsed).strip()
    except Exception:
        return re.sub(r"^\[?['\"]?|['\"]?\]?$", "", str(raw).strip()).strip()


def sent_split(text):
    if not text or not text.strip():
        return []
    text = re.sub(r"\n{2,}", " ", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"(?<!\w)\d+[\.\)]\s+", " ", text)   # strip leading clause numbers
    text = re.sub(r"\s+", " ", text).strip()
    raw  = re.split(r'(?<=[.?!])\s+(?=[A-Z"(])', text)
    merged, buf = [], ""
    for s in raw:
        buf = (buf + " " + s).strip() if buf else s.strip()
        if len(buf) >= 30:
            merged.append(buf)
            buf = ""
    if buf:
        if merged:
            merged[-1] = merged[-1] + " " + buf
        else:
            merged.append(buf)
    final = []
    for s in merged:
        if len(s) > 500:
            parts = re.split(r"[;]\s+", s)
            final.extend(p.strip() for p in parts if len(p.strip().split()) >= 8)
        elif len(s.split()) >= 8:
            final.append(s.strip())
    return final


# ── BIO tagging via regex span detection ───────────────────────────────────

def _token_offsets(tokens):
    offsets, pos = [], 0
    for tok in tokens:
        offsets.append((pos, pos + len(tok)))
        pos += len(tok) + 1
    return offsets


def _pick_closest(matches, anchor_iso):
    try:
        anchor_dt  = dateutil_parser.parse(anchor_iso)
        default_dt = datetime(anchor_dt.year, 1, 1)
    except Exception:
        return matches[0]
    best, best_diff = matches[0], float("inf")
    for m in matches:
        try:
            m_dt = dateutil_parser.parse(m.group(), default=default_dt)
            diff = abs((m_dt - anchor_dt).days)
            if diff < best_diff:
                best, best_diff = m, diff
        except Exception:
            continue
    return best


def _apply_bio(tags, offsets, span_s, span_e, b_tag, i_tag):
    first = True
    for idx, (ts, te) in enumerate(offsets):
        if te <= span_s or ts >= span_e:
            continue
        tags[idx] = b_tag if first else i_tag
        first = False


def bio_tag_regex(tokens, ctype, anchor_iso=None):
    """Regex-based BIO tagging. Replaces the old word-list heuristic."""
    if ctype == "none":
        return ["O"] * len(tokens)

    text    = " ".join(tokens)
    offsets = _token_offsets(tokens)
    tags    = ["O"] * len(tokens)

    if ctype in ("expiration", "effective"):
        b_tag = "B-EXP_DATE"   if ctype == "expiration" else "B-START_DATE"
        i_tag = "I-EXP_DATE"   if ctype == "expiration" else "I-START_DATE"
        matches = list(DATE_RE.finditer(text))
        if not matches:
            return tags
        if anchor_iso and len(matches) > 1:
            best = _pick_closest(matches, anchor_iso)
        else:
            non_year = [m for m in matches
                        if not re.fullmatch(r"(?:19|20)\d{2}", m.group())]
            best = non_year[0] if non_year else matches[0]
        _apply_bio(tags, offsets, best.start(), best.end(), b_tag, i_tag)

    elif ctype == "duration":
        matches = list(DURATION_RE.finditer(text))
        if matches:
            for m in matches:
                _apply_bio(tags, offsets, m.start(), m.end(), "B-DURATION", "I-DURATION")
        else:
            date_matches = list(DATE_RE.finditer(text))
            if date_matches:
                non_year = [m for m in date_matches
                            if not re.fullmatch(r"(?:19|20)\d{2}", m.group())]
                best = non_year[0] if non_year else date_matches[0]
                _apply_bio(tags, offsets, best.start(), best.end(), "B-START_DATE", "I-START_DATE")

    return tags


# ── Contract processing — hybrid strategy ──────────────────────────────────

def process_contract(row, split_name):
    exp_clause  = normalise(clean_clause(row.get("Expiration Date",  "") or ""))
    exp_iso     = row.get("expiration_date_iso") or None
    eff_clause  = normalise(clean_clause(row.get("Effective Date",   "") or ""))
    eff_iso     = row.get("effective_date_iso")  or None
    ren_clause  = normalise(clean_clause(row.get("Renewal Term",     "") or ""))
    ocr         = normalise(row.get("ocr_text", "") or "")
    contract_id = row.get("Filename", "")
    examples    = []
    seen_sents  = set()

    def add(sent, clf, ctype, gt):
        tokens = sent.split()
        if len(tokens) < 8 or sent in seen_sents:
            return
        seen_sents.add(sent)
        anchor   = gt if (ctype in ("expiration", "effective") and gt) else None
        ner_tags = [NER_L2I[t] for t in bio_tag_regex(tokens, ctype, anchor)]
        examples.append({
            "contract_id":       contract_id,
            "split":             split_name,
            "sentence":          sent,
            "tokens":            tokens,
            "ner_tags":          ner_tags,
            "classifier_label":  clf,
            "ground_truth_date": gt,
        })

    # ── Positive examples: directly from clause columns (no overlap guessing)
    for sent in sent_split(exp_clause):
        add(sent, 1, "expiration", exp_iso or "")
    for sent in sent_split(eff_clause):
        add(sent, 2, "effective",  eff_iso or "")
    for sent in sent_split(ren_clause):
        add(sent, 3, "duration",   "")

    # ── None examples: OCR sentences not in any clause (matches inference distribution)
    all_clause_words = set(
        (exp_clause + " " + eff_clause + " " + ren_clause).lower().split()
    )
    for sent in sent_split(ocr):
        if sent in seen_sents:
            continue
        if all_clause_words:
            sw      = set(sent.lower().split())
            overlap = len(sw & all_clause_words) / max(len(sw), 1)
            if overlap >= 0.3:
                continue
        add(sent, 0, "none", "")

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
    print("Contracts — train:", len(raw["train"]),
          "| val:",  len(raw["val"]),
          "| test:", len(raw["test"]))

    train_ds = build_split(raw["train"], "train")
    val_ds   = build_split(raw["val"],   "val")
    test_ds  = build_split(raw["test"],  "test")

    dd = DatasetDict({"train": train_ds, "val": val_ds, "test": test_ds})

    print("\n=== Sentence-level dataset ===")
    for name, ds in dd.items():
        c     = Counter(ds["classifier_label"])
        total = len(ds)
        print(f"  [{name}] {total:>6} sentences | "
              f"none={c[0]:>5} ({100*c[0]/max(total,1):.1f}%)  "
              f"expiration={c[1]:>4} ({100*c[1]/max(total,1):.1f}%)  "
              f"effective={c[2]:>4} ({100*c[2]/max(total,1):.1f}%)  "
              f"renewal={c[3]:>4} ({100*c[3]/max(total,1):.1f}%)")
        ner_counts = Counter()
        for tags in ds["ner_tags"]:
            ner_counts.update(tags)
        print(f"         NER tags: { {NER_LABELS[k]: v for k, v in sorted(ner_counts.items())} }")

    os.makedirs(args.save_path, exist_ok=True)
    dd.save_to_disk(args.save_path)

    os.makedirs("./data", exist_ok=True)
    with open("./data/label_maps.json", "w") as f:
        json.dump({
            "ner_label2id": NER_L2I,
            "ner_id2label": {str(k): v for k, v in enumerate(NER_LABELS)},
            "clf_label2id": CLF_L2I,
            "clf_id2label": {str(k): v for k, v in CLF_I2L.items()},
            "save_path":    args.save_path,
        }, f, indent=2)

    print(f"\nSaved to {args.save_path}")
    print("Label maps saved to ./data/label_maps.json")


if __name__ == "__main__":
    main()
