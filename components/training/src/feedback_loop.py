"""
Confidence-gated feedback loop.

Modes:
  --collect   Read predict.py output JSON; append uncertain events to queue file.
  --retrain   Read human-labelled queue entries; append to training data; trigger retrain.
  --status    Print queue size and distance to retrain trigger.
"""
import os, json, argparse, subprocess
from datetime import datetime

QUEUE_FILE           = "./data/uncertain_samples.jsonl"
CONFIDENCE_THRESHOLD = 0.7
RETRAIN_TRIGGER_N    = 100

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://129.114.27.190:30500")
EXPERIMENT = "deadline-feedback"

os.environ["AWS_ACCESS_KEY_ID"]      = os.getenv("AWS_ACCESS_KEY_ID", "datanauts-key")
os.environ["AWS_SECRET_ACCESS_KEY"]  = os.getenv("AWS_SECRET_ACCESS_KEY", "datanauts-secret")
os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://129.114.27.190:30900")


def _queue_size():
    if not os.path.exists(QUEUE_FILE):
        return 0
    with open(QUEUE_FILE) as f:
        return sum(1 for _ in f)


def _log_to_mlflow(entries):
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment(EXPERIMENT)
        with mlflow.start_run(run_name=f"feedback_{datetime.utcnow():%Y%m%d_%H%M%S}"):
            mlflow.set_tag("uncertain", "true")
            mlflow.log_metric("new_uncertain_samples", len(entries))
            mlflow.log_metric("queue_total", _queue_size())
    except Exception as e:
        print(f"[feedback] MLflow log failed: {e}")


def collect(predictions_path, threshold, queue_file, trigger_n):
    with open(predictions_path) as f:
        result = json.load(f)

    uncertain_events = [
        e for e in result.get("events", []) if e.get("uncertain", False)
                or e.get("confidence", 1.0) < threshold
    ]

    if not uncertain_events:
        print("[feedback] No uncertain events in this prediction.")
        return

    os.makedirs(os.path.dirname(os.path.abspath(queue_file)), exist_ok=True)
    new_entries = []
    with open(queue_file, "a") as out:
        for event in uncertain_events:
            entry = {
                "contract_id":      result.get("contract_id", "unknown"),
                "sentence":         event.get("source_sentence", ""),
                "predicted_label":  event.get("event_type", ""),
                "confidence":       event.get("confidence", 0),
                "class_scores":     event.get("class_scores", {}),
                "timestamp":        datetime.utcnow().isoformat(),
                "human_label":      None,
            }
            out.write(json.dumps(entry) + "\n")
            new_entries.append(entry)

    q_size = _queue_size()
    print(f"[feedback] Added {len(new_entries)} uncertain events → queue: {q_size} / {trigger_n}")

    _log_to_mlflow(new_entries)

    if q_size >= trigger_n:
        print(f"\n[ALERT] Queue reached {q_size} samples — RETRAIN TRIGGER reached!")
        print(f"  Next step: add human_label to {queue_file}, then run:")
        print(f"  python src/feedback_loop.py --retrain --clf_model roberta_clf_v5 --next_version v7\n")


def retrain(clf_model, next_version, queue_file):
    if not os.path.exists(queue_file):
        print(f"[feedback] Queue file not found: {queue_file}")
        return

    labelled = []
    with open(queue_file) as f:
        for line in f:
            entry = json.loads(line.strip())
            if entry.get("human_label") is not None:
                labelled.append(entry)

    if not labelled:
        print("[feedback] No human-labelled entries found in queue. Add 'human_label' to entries first.")
        return

    print(f"[feedback] Found {len(labelled)} labelled entries.")

    additions_path = "./data/feedback_additions.jsonl"
    with open(additions_path, "w") as f:
        for e in labelled:
            f.write(json.dumps({
                "sentence":          e["sentence"],
                "classifier_label":  e["human_label"],
                "contract_id":       e["contract_id"],
                "split":             "train",
                "ground_truth_date": "",
                "tokens":            e["sentence"].split(),
                "ner_tags":          [0] * len(e["sentence"].split()),
            }) + "\n")

    print(f"[feedback] Feedback training samples written to {additions_path}")
    print(f"[feedback] Triggering retrain: roberta_clf_{next_version}")

    result = subprocess.run(
        ["python", "src/train_classifier.py", "--model", f"roberta_clf_{next_version}"],
        capture_output=False,
    )
    if result.returncode == 0:
        print(f"[feedback] Retrain completed successfully.")
        archive_path = queue_file.replace(".jsonl", f"_archived_{datetime.utcnow():%Y%m%d}.jsonl")
        os.rename(queue_file, archive_path)
        print(f"[feedback] Queue archived to {archive_path}")
    else:
        print(f"[feedback] Retrain failed with exit code {result.returncode}")


def status(queue_file, trigger_n):
    q_size = _queue_size()
    remaining = max(trigger_n - q_size, 0)
    print(f"\n[feedback] Queue status")
    print(f"  File            : {queue_file}")
    print(f"  Total entries   : {q_size}")
    print(f"  Trigger at      : {trigger_n}")
    print(f"  Remaining       : {remaining}")
    if q_size >= trigger_n:
        print(f"  STATUS          : *** RETRAIN TRIGGER REACHED ***")
    else:
        print(f"  STATUS          : collecting ({remaining} more needed)")

    if os.path.exists(queue_file):
        labelled = 0
        with open(queue_file) as f:
            for line in f:
                try:
                    if json.loads(line.strip()).get("human_label") is not None:
                        labelled += 1
                except Exception:
                    pass
        print(f"  Human-labelled  : {labelled} / {q_size}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Feedback loop for deadline detection.")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--collect",  action="store_true", help="Collect uncertain predictions into queue")
    group.add_argument("--retrain",  action="store_true", help="Trigger retrain from labelled queue")
    group.add_argument("--status",   action="store_true", help="Print queue size and trigger status")

    parser.add_argument("--predictions", default=None,      help="[collect] Path to predict.py output JSON")
    parser.add_argument("--clf_model",   default="roberta_clf_v5", help="[retrain] Current classifier model name")
    parser.add_argument("--next_version",default="v7",      help="[retrain] Version tag for retrained model")
    parser.add_argument("--threshold",   type=float, default=CONFIDENCE_THRESHOLD)
    parser.add_argument("--trigger_n",   type=int,   default=RETRAIN_TRIGGER_N)
    parser.add_argument("--queue_file",  default=QUEUE_FILE)
    args = parser.parse_args()

    if args.collect:
        if not args.predictions:
            parser.error("--collect requires --predictions <path_to_predict_output.json>")
        collect(args.predictions, args.threshold, args.queue_file, args.trigger_n)
    elif args.retrain:
        retrain(args.clf_model, args.next_version, args.queue_file)
    elif args.status:
        status(args.queue_file, args.trigger_n)


if __name__ == "__main__":
    main()
