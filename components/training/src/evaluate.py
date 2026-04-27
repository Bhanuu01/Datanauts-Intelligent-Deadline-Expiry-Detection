import os, json, argparse
from collections import defaultdict
from dateutil import parser as dateutil_parser
from datasets import load_dataset, load_from_disk
import mlflow

from predict import predict

MLFLOW_URI  = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow.platform.svc.cluster.local:5000")
EXPERIMENT  = "deadline-detection-e2e"
DATASET_ID  = "tanvitakavane/datanauts_project_cuad-deadline-ner-version2"
DATA_PATH   = "./data/deadline_sentences"
DATE_WINDOW = 30   # days — "within N days" match

os.environ["AWS_ACCESS_KEY_ID"]      = os.getenv("AWS_ACCESS_KEY_ID", "datanauts-key")
os.environ["AWS_SECRET_ACCESS_KEY"]  = os.getenv("AWS_SECRET_ACCESS_KEY", "datanauts-secret")
os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv(
    "MLFLOW_S3_ENDPOINT_URL", "http://minio.platform.svc.cluster.local:9000"
)


def get_optional_limit(name):
    raw = os.getenv(name, "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer if set") from exc
    return value if value > 0 else None


def limited_test_views(raw_test_contracts, dd_test_rows):
    eval_contract_limit = get_optional_limit("BOOTSTRAP_MAX_EVAL_CONTRACTS")
    eval_sentence_limit = get_optional_limit("BOOTSTRAP_MAX_TEST_SAMPLES")

    if not eval_contract_limit and not eval_sentence_limit:
        return raw_test_contracts, dd_test_rows

    limited_rows = dd_test_rows
    if eval_sentence_limit:
        limited_rows = dd_test_rows.select(range(min(eval_sentence_limit, len(dd_test_rows))))

    allowed_contract_ids = []
    seen_ids = set()
    for row in limited_rows:
        contract_id = row.get("contract_id", "")
        if contract_id and contract_id not in seen_ids:
            seen_ids.add(contract_id)
            allowed_contract_ids.append(contract_id)

    if eval_contract_limit:
        allowed_contract_ids = allowed_contract_ids[:eval_contract_limit]

    allowed_contract_ids_set = set(allowed_contract_ids)
    limited_contracts = raw_test_contracts.filter(
        lambda record: record.get("Filename", "") in allowed_contract_ids_set
    )
    limited_rows = limited_rows.filter(
        lambda row: row.get("contract_id", "") in allowed_contract_ids_set
    )
    print(
        "[INFO] Limiting end-to-end evaluation to "
        f"{len(limited_contracts)} contracts / {len(limited_rows)} sentences "
        "via BOOTSTRAP_MAX_EVAL_CONTRACTS/BOOTSTRAP_MAX_TEST_SAMPLES"
    )
    return limited_contracts, limited_rows


def dates_within(pred_date, gt_date, window=DATE_WINDOW):
    try:
        pd = dateutil_parser.parse(pred_date)
        gd = dateutil_parser.parse(gt_date)
        return abs((pd - gd).days) <= window
    except Exception:
        return False


def run_evaluation(clf_model_path, ner_model_path, threshold=0.7, candidate_latency_p95_ms=0.0):
    dd_disk  = load_from_disk(DATA_PATH)
    raw_hf   = load_dataset(DATASET_ID)

    test_contracts, dd_test = limited_test_views(raw_hf["test"], dd_disk["test"])

    stats = {
        "total":                0,
        "with_gt_date":         0,    # contracts that have at least one GT date
        "covered":              0,    # GT contracts where we returned any event
        "exact_match":          0,    # predicted date == GT date (exact string)
        "within_30_days":       0,    # predicted date within 30 days of GT
        "true_none":            0,    # no GT date, no prediction
        "false_alarm":          0,    # no GT date but we predicted deadline
        "uncertain_flagged":    0,    # at least one uncertain event returned
        "multi_date_conflict":  0,    # at least one event with conflicting date candidates
    }
    per_type = defaultdict(lambda: {"covered": 0, "exact": 0, "within_30": 0, "total_gt": 0})

    test_sentences_by_contract = defaultdict(list)
    for row in dd_test:
        test_sentences_by_contract[row["contract_id"]].append({
            "sentence":         row["sentence"],
            "classifier_label": int(row["classifier_label"]),
            "ground_truth_date": row.get("ground_truth_date", ""),
        })

    for contract in test_contracts:
        cid      = contract.get("Filename", "")
        exp_iso  = contract.get("expiration_date_iso") or ""
        eff_iso  = contract.get("effective_date_iso")  or ""
        agr_iso  = contract.get("agreement_date_iso")  or ""
        gt_dates = {}
        if exp_iso:
            gt_dates["expiration"] = exp_iso
        if eff_iso:
            gt_dates["effective"]  = eff_iso
        if agr_iso:
            gt_dates["agreement"]  = agr_iso

        sents = [r["sentence"] for r in test_sentences_by_contract.get(cid, [])]
        if not sents:
            continue

        stats["total"] += 1
        has_gt = bool(gt_dates)
        if has_gt:
            stats["with_gt_date"] += 1

        result = predict(
            sentences=sents,
            clf_model_path=clf_model_path,
            ner_model_path=ner_model_path,
            contract_id=cid,
            confidence_threshold=threshold,
        )

        if result.get("uncertain"):
            stats["uncertain_flagged"] += 1
        if result.get("multi_date_conflict"):
            stats["multi_date_conflict"] += 1

        if not has_gt:
            if not result["has_deadline"]:
                stats["true_none"] += 1
            else:
                stats["false_alarm"] += 1
            continue

        pred_events = {e["event_type"]: e for e in result.get("events", [])}
        any_covered = False

        for event_type, gt_date in gt_dates.items():
            per_type[event_type]["total_gt"] += 1
            if event_type not in pred_events:
                continue
            per_type[event_type]["covered"] += 1
            any_covered = True
            pred_date   = pred_events[event_type].get("deadline_date") or ""
            if pred_date:
                if pred_date == gt_date:
                    per_type[event_type]["exact"] += 1
                    stats["exact_match"] += 1
                if dates_within(pred_date, gt_date, DATE_WINDOW):
                    per_type[event_type]["within_30"] += 1
                    stats["within_30_days"] += 1

        if any_covered:
            stats["covered"] += 1

    n_gt = max(stats["with_gt_date"], 1)
    print("\n========= End-to-End Evaluation =========")
    print(f"  Total test contracts   : {stats['total']}")
    print(f"  With GT date           : {stats['with_gt_date']}")
    print(f"  Coverage (% of GT)     : {100*stats['covered']/n_gt:.1f}%  ({stats['covered']}/{n_gt})")
    print(f"  Exact match            : {100*stats['exact_match']/n_gt:.1f}%")
    print(f"  Within {DATE_WINDOW} days          : {100*stats['within_30_days']/n_gt:.1f}%")
    print(f"  True none (correct)    : {stats['true_none']}")
    print(f"  False alarms           : {stats['false_alarm']}")
    print(f"  Uncertain flagged      : {stats['uncertain_flagged']}")
    print(f"  Multi-date conflicts   : {stats['multi_date_conflict']}")
    print()
    for et, m in per_type.items():
        tot = max(m["total_gt"], 1)
        print(f"  [{et}]  covered={100*m['covered']/tot:.0f}%  exact={100*m['exact']/tot:.0f}%  within30={100*m['within_30']/tot:.0f}%")
    print("=========================================\n")

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)
    run_name = f"e2e_{os.path.basename(clf_model_path)}_{os.path.basename(ner_model_path)}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "clf_model":  clf_model_path,
            "ner_model":  ner_model_path,
            "threshold":  threshold,
            "date_window": DATE_WINDOW,
        })
        mlflow.log_metrics({
            "coverage_pct":          100 * stats["covered"]            / n_gt,
            "exact_match_pct":       100 * stats["exact_match"]        / n_gt,
            "within_30_days_pct":    100 * stats["within_30_days"]     / n_gt,
            "false_alarm_count":     stats["false_alarm"],
            "true_none_count":       stats["true_none"],
            "uncertain_flagged":     stats["uncertain_flagged"],
            "multi_date_conflict":   stats["multi_date_conflict"],
            "total_test_contracts":  stats["total"],
        })
        for et, m in per_type.items():
            tot = max(m["total_gt"], 1)
            mlflow.log_metrics({
                f"{et}_coverage_pct":    100 * m["covered"]   / tot,
                f"{et}_exact_match_pct": 100 * m["exact"]     / tot,
                f"{et}_within30_pct":    100 * m["within_30"] / tot,
            })
        print(f"E2E metrics logged → {MLFLOW_URI} | experiment: {EXPERIMENT}")

    summary = {
        "ner_f1": 0.0,
        "clf_macro_f1": 0.0,
        "e2e_coverage": stats["covered"] / n_gt,
        "false_alarm_count": stats["false_alarm"],
        "candidate_latency_p95_ms": candidate_latency_p95_ms,
        "max_latency_p95_ms": float(os.getenv("MAX_LATENCY_P95_MS", "500")),
        "coverage_pct": 100 * stats["covered"] / n_gt,
        "exact_match_pct": 100 * stats["exact_match"] / n_gt,
        "within_30_days_pct": 100 * stats["within_30_days"] / n_gt,
        "true_none_count": stats["true_none"],
        "uncertain_flagged": stats["uncertain_flagged"],
        "multi_date_conflict": stats["multi_date_conflict"],
        "total_test_contracts": stats["total"],
        "per_type": dict(per_type),
    }

    return stats, dict(per_type), summary


def run_cross_domain_evaluation(samples_path, clf_model_path, ner_model_path, threshold=0.7):
    """
    Evaluate classifier generalization on held-out cross-domain samples.

    samples_path: path to a JSON file with a list of:
        [{"sentence": str, "expected_label": str, "source": str}, ...]

    Reports per-class accuracy and logs results to MLflow under
    experiment 'deadline-detection-cross-domain'.
    """
    import json
    from transformers import pipeline as hf_pipeline
    from sklearn.metrics import classification_report as skr

    with open(samples_path) as f:
        samples = json.load(f)

    clf_pipe = hf_pipeline(
        "text-classification",
        model=clf_model_path,
        top_k=None,
        device=-1,
    )

    labels_true, labels_pred, sources = [], [], []
    for item in samples:
        results = clf_pipe(item["sentence"])[0]
        best    = max(results, key=lambda x: x["score"])
        labels_true.append(item["expected_label"])
        labels_pred.append(best["label"].lower())
        sources.append(item.get("source", "unknown"))

    all_labels = sorted(set(labels_true + labels_pred))
    report     = skr(labels_true, labels_pred, labels=all_labels, zero_division=0)
    n          = len(samples)
    correct    = sum(t == p for t, p in zip(labels_true, labels_pred))

    print("\n======= Cross-Domain Evaluation =========")
    print(f"  Samples          : {n}")
    print(f"  Overall accuracy : {100*correct/max(n,1):.1f}%")
    print(report)
    print("=========================================")

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("deadline-detection-cross-domain")
    with mlflow.start_run(run_name=f"cross_domain_{os.path.basename(clf_model_path)}"):
        mlflow.log_params({"clf_model": clf_model_path, "samples_path": samples_path, "n_samples": n})
        mlflow.log_metric("cross_domain_accuracy", correct / max(n, 1))
        mlflow.log_text(report, "cross_domain_report.txt")
        print(f"Cross-domain metrics logged → {MLFLOW_URI}")

    return {"accuracy": correct / max(n, 1), "n": n}


def main():
    parser = argparse.ArgumentParser(description="End-to-end deadline detection evaluation")
    parser.add_argument("--clf_model", required=True)
    parser.add_argument("--ner_model", required=True)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument("--candidate_latency_p95_ms", type=float, default=0.0)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--cross_domain_samples", default=None,
                        help="Optional path to cross-domain samples JSON for generalization test")
    args = parser.parse_args()

    _, _, summary = run_evaluation(
        args.clf_model,
        args.ner_model,
        args.threshold,
        args.candidate_latency_p95_ms,
    )
    if args.cross_domain_samples:
        cross_domain = run_cross_domain_evaluation(
            args.cross_domain_samples, args.clf_model, args.ner_model, args.threshold
        )
        summary["cross_domain_accuracy"] = cross_domain["accuracy"]
        summary["cross_domain_samples"] = cross_domain["n"]

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
