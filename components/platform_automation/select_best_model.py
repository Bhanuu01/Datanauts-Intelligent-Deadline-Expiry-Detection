"""
select_best_model.py — Query MLflow for the best NER and classifier runs by
test_f1, download their artifacts to local paths, and write a model_selection.json
that the rest of the pipeline (run_retrain_cycle, evaluate_and_promote) can consume.

Usage:
    python select_best_model.py [--output /tmp/model_selection.json] [--download-dir /tmp/best-models]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

MLFLOW_URI  = os.getenv("MLFLOW_TRACKING_URI",   "http://127.0.0.1:8000")
NER_EXP     = os.getenv("NER_EXPERIMENT_NAME",   "deadline-detection-ner")
CLF_EXP     = os.getenv("CLASSIFIER_EXPERIMENT_NAME", "deadline-detection-classifier")
METRIC_KEY  = os.getenv("SELECTION_METRIC",      "test_f1")   # override with e.g. "test_EXP_DATE_f1"


def best_run(client: MlflowClient, experiment_name: str, metric: str = METRIC_KEY):
    """Return the run with the highest value of `metric` in the given experiment."""
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise RuntimeError(f"MLflow experiment not found: {experiment_name!r}")

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=f"metrics.{metric} > 0 and params.epochs != '0'",  # exclude baselines
        order_by=[f"metrics.{metric} DESC"],
        max_results=1,
    )
    if not runs:
        # Fallback: no trained run found — try any run with the metric
        print(f"[WARN] No trained runs with metric '{metric}' in {experiment_name!r}. "
              f"Falling back to most-recent run.", file=sys.stderr)
        runs = client.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
    if not runs:
        raise RuntimeError(f"No completed runs found in experiment: {experiment_name!r}")

    run = runs[0]
    return run


def download_artifacts(client: MlflowClient, run_id: str, artifact_subpath: str,
                       local_dir: Path) -> Path:
    """
    Download artifacts from an MLflow run.
    artifact_subpath: the artifact subfolder logged (typically 'model').
    Returns the local path where artifacts were downloaded.
    """
    local_dir.mkdir(parents=True, exist_ok=True)
    dst = client.download_artifacts(run_id, artifact_subpath, str(local_dir))
    return Path(dst)


def main():
    parser = argparse.ArgumentParser(description="Select best NER + CLF models from MLflow by test_f1.")
    parser.add_argument("--output",       default="/tmp/model_selection.json",
                        help="Path to write the model_selection.json result.")
    parser.add_argument("--download-dir", default="/tmp/best-models",
                        help="Root directory to download model artifacts into.")
    parser.add_argument("--metric",       default=METRIC_KEY,
                        help=f"MLflow metric key to rank by (default: {METRIC_KEY}).")
    parser.add_argument("--no-download",  action="store_true",
                        help="Print selection info only — do not download artifacts.")
    args = parser.parse_args()

    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient(tracking_uri=MLFLOW_URI)
    download_root = Path(args.download_dir)

    print(f"\n=== MLflow best-model selection (metric: {args.metric}) ===")
    print(f"    Tracking URI : {MLFLOW_URI}")
    print(f"    NER experiment  : {NER_EXP}")
    print(f"    CLF experiment  : {CLF_EXP}\n")

    # ── NER ──────────────────────────────────────────────────────────────────
    ner_run = best_run(client, NER_EXP, args.metric)
    ner_f1  = ner_run.data.metrics.get(args.metric, 0.0)
    ner_model_name = ner_run.data.params.get("model_name", ner_run.data.params.get("model", "unknown"))
    print(f"[NER]  Best run  : {ner_run.info.run_id}")
    print(f"       Model name: {ner_model_name}")
    print(f"       {args.metric}: {ner_f1:.4f}")
    print(f"       All metrics: { {k: round(v,4) for k,v in ner_run.data.metrics.items()} }")

    # ── CLF ──────────────────────────────────────────────────────────────────
    clf_run = best_run(client, CLF_EXP, args.metric)
    clf_f1  = clf_run.data.metrics.get(args.metric, 0.0)
    clf_model_name = clf_run.data.params.get("model_name", clf_run.data.params.get("model", "unknown"))
    print(f"\n[CLF]  Best run  : {clf_run.info.run_id}")
    print(f"       Model name: {clf_model_name}")
    print(f"       {args.metric}: {clf_f1:.4f}")
    print(f"       All metrics: { {k: round(v,4) for k,v in clf_run.data.metrics.items()} }")

    # ── Download ─────────────────────────────────────────────────────────────
    ner_local_path = str(download_root / "ner")
    clf_local_path = str(download_root / "classifier")

    if not args.no_download:
        print(f"\n[NER]  Downloading artifacts → {ner_local_path} ...")
        try:
            ner_local_path = str(download_artifacts(
                client, ner_run.info.run_id, "model", download_root / "ner"
            ))
            print(f"[NER]  Downloaded to: {ner_local_path}")
        except Exception as e:
            print(f"[NER]  Artifact download failed: {e}. Path will be empty.", file=sys.stderr)

        print(f"[CLF]  Downloading artifacts → {clf_local_path} ...")
        try:
            clf_local_path = str(download_artifacts(
                client, clf_run.info.run_id, "model", download_root / "classifier"
            ))
            print(f"[CLF]  Downloaded to: {clf_local_path}")
        except Exception as e:
            print(f"[CLF]  Artifact download failed: {e}. Path will be empty.", file=sys.stderr)

    # ── Write selection JSON ──────────────────────────────────────────────────
    selection = {
        "selection_metric": args.metric,
        "ner": {
            "run_id":     ner_run.info.run_id,
            "model_name": ner_model_name,
            "metrics":    dict(ner_run.data.metrics),
            "params":     dict(ner_run.data.params),
            "local_path": ner_local_path,
        },
        "classifier": {
            "run_id":     clf_run.info.run_id,
            "model_name": clf_model_name,
            "metrics":    dict(clf_run.data.metrics),
            "params":     dict(clf_run.data.params),
            "local_path": clf_local_path,
        },
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(selection, indent=2))
    print(f"\n[OK]  Selection written to: {out}")
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
