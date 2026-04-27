import json
import os
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List

from components.common.object_store import download_json
from components.platform_automation.feedback_curation import compile_feedback_training_additions
from mlflow.tracking import MlflowClient
from components.common.object_store import load_json_objects
from components.common.object_store import object_store_enabled
from components.common.object_store import upload_directory_as_tarball
from components.common.object_store import upload_json


DEFAULT_RELEASE_ROOT = "/tmp/model-releases"
DEFAULT_METRICS_PATH = "/tmp/feedback_metrics.json"
DEFAULT_OUTPUT_PATH = "/tmp/retrain_decision.json"
DEFAULT_EVAL_PATH = "/tmp/evaluation_metrics.json"
DEFAULT_PAPERLESS_URL = "http://paperless-ngx.paperless.svc.cluster.local:8000"
DEFAULT_FEEDBACK_CHECKPOINT_PATH = "/tmp/feedback_checkpoint.json"
DEFAULT_NER_REGISTERED_MODEL = "deadline-ner"
DEFAULT_CLASSIFIER_REGISTERED_MODEL = "deadline-classifier"
DEFAULT_MODEL_CANDIDATE_PAIRS = ",".join([
    "bert_ner_v4:roberta_clf_v3",
    "contracts_bert_ner_v1:deberta_clf_v1",
    "legal_bert_ner_v1:deberta_clf_v1",
])
DEFAULT_RUNTIME_LOG_BUCKET = "datanauts-runtime"
DEFAULT_MODEL_ARTIFACT_BUCKET = "datanauts-models"
DEFAULT_CURRENT_PRODUCTION_METRICS_PATH = "/tmp/current_production_metrics.json"


def paperless_auth_header() -> str | None:
    password = os.getenv("PAPERLESS_ADMIN_PASSWORD")
    if not password:
        return None

    import base64

    username = os.getenv("PAPERLESS_ADMIN_USER", "admin")
    token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
    return f"Basic {token}"


def paperless_request_json(url: str) -> Dict[str, Any]:
    headers = {"Accept": "application/json"}
    auth_header = paperless_auth_header()
    if auth_header:
        headers["Authorization"] = auth_header
    request = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_paperless_tag_count(tag_name: str) -> int:
    auth_header = paperless_auth_header()
    if not auth_header:
        return 0

    paperless_url = os.getenv("PAPERLESS_API_URL", DEFAULT_PAPERLESS_URL)
    query = urllib.parse.urlencode({"name__iexact": tag_name, "page_size": 1})
    tag_response = paperless_request_json(f"{paperless_url}/api/tags/?{query}")
    tag_results = tag_response.get("results", [])
    if not tag_results:
        return 0

    tag_id = tag_results[0]["id"]
    document_query = urllib.parse.urlencode({"tags__id__all": tag_id, "page_size": 1})
    document_response = paperless_request_json(f"{paperless_url}/api/documents/?{document_query}")
    return int(document_response.get("count", 0))


def maybe_upload_json(path_key_env: str, default_key: str, payload: Dict[str, Any]) -> None:
    if not object_store_enabled():
        return
    upload_json(
        bucket=os.getenv("RUNTIME_LOG_BUCKET", DEFAULT_RUNTIME_LOG_BUCKET),
        key=os.getenv(path_key_env, default_key),
        payload=payload,
    )


def load_feedback_metrics() -> Dict[str, Any]:
    metrics_path = Path(os.getenv("FEEDBACK_METRICS_PATH", DEFAULT_METRICS_PATH))
    if metrics_path.exists():
        return json.loads(metrics_path.read_text())

    feedback_log_path = Path(os.getenv("FEEDBACK_LOG_PATH", "/tmp/feedback_events.jsonl"))
    feedback_prefix = os.getenv("FEEDBACK_S3_PREFIX", "runtime/online-features/feedback")
    all_lines: List[Dict[str, Any]] = []
    if object_store_enabled():
        try:
            all_lines = load_json_objects(
                os.getenv("RUNTIME_LOG_BUCKET", DEFAULT_RUNTIME_LOG_BUCKET),
                feedback_prefix,
            )
        except Exception:
            all_lines = []
    elif feedback_log_path.exists():
        all_lines = [json.loads(line) for line in feedback_log_path.read_text().splitlines() if line.strip()]

    if all_lines:
        checkpoint_path = Path(os.getenv("FEEDBACK_CHECKPOINT_PATH", DEFAULT_FEEDBACK_CHECKPOINT_PATH))
        last_trained_count = 0
        if checkpoint_path.exists():
            try:
                checkpoint = json.loads(checkpoint_path.read_text())
                last_trained_count = int(checkpoint.get("last_trained_object_count", checkpoint.get("last_trained_line_count", 0)))
            except (json.JSONDecodeError, KeyError, ValueError):
                pass

        lines = all_lines[last_trained_count:]
        feedback_events = len(lines)
        correction_events = sum(
            1 for line in lines if line.get("event") in {"dismiss", "edit", "manual_add"}
        )
        correction_rate = (correction_events / feedback_events) if feedback_events else 0.0
        wrong_feedback_selections = fetch_paperless_tag_count("Action:Reject")
        correct_feedback_selections = fetch_paperless_tag_count("Action:Accept")
        return {
            "new_feedback_events": feedback_events,
            "correction_rate_7d": correction_rate,
            "wrong_feedback_selections": wrong_feedback_selections,
            "correct_feedback_selections": correct_feedback_selections,
            "_total_feedback_events": len(all_lines),
            "_last_trained_object_count": last_trained_count,
        }

    return {
        "new_feedback_events": int(os.getenv("NEW_FEEDBACK_EVENTS", "0")),
        "correction_rate_7d": float(os.getenv("CORRECTION_RATE_7D", "0.0")),
        "wrong_feedback_selections": int(os.getenv("WRONG_FEEDBACK_SELECTIONS", "0")),
        "correct_feedback_selections": int(os.getenv("CORRECT_FEEDBACK_SELECTIONS", "0")),
    }


def should_retrain(metrics: Dict[str, Any]) -> Dict[str, Any]:
    min_feedback = int(os.getenv("MIN_FEEDBACK_EVENTS", "500"))
    max_correction_rate = float(os.getenv("MAX_CORRECTION_RATE", "0.15"))
    min_wrong_feedback = int(os.getenv("MIN_WRONG_FEEDBACK_SELECTIONS", "50"))

    feedback_count = int(metrics.get("new_feedback_events", 0))
    correction_rate = float(metrics.get("correction_rate_7d", 0.0))
    wrong_feedback = int(metrics.get("wrong_feedback_selections", 0))
    correct_feedback = int(metrics.get("correct_feedback_selections", 0))

    reasons: List[str] = []
    if feedback_count >= min_feedback:
        reasons.append("feedback_volume")
    if correction_rate >= max_correction_rate:
        reasons.append("correction_rate")
    if wrong_feedback >= min_wrong_feedback:
        reasons.append("wrong_feedback_selections")

    return {
        "trigger_retrain": bool(reasons),
        "reasons": reasons,
        "metrics": {
            "new_feedback_events": feedback_count,
            "correction_rate_7d": correction_rate,
            "wrong_feedback_selections": wrong_feedback,
            "correct_feedback_selections": correct_feedback,
        },
        "thresholds": {
            "min_feedback_events": min_feedback,
            "max_correction_rate": max_correction_rate,
            "min_wrong_feedback_selections": min_wrong_feedback,
        },
    }


def run_command(command: List[str], env: Dict[str, str] | None = None) -> int:
    process = subprocess.run(command, check=False, env=env)
    return process.returncode


def parse_model_candidate_pairs(default_ner_model: str, default_classifier_model: str) -> List[Dict[str, str]]:
    raw_pairs = os.getenv("MODEL_CANDIDATE_PAIRS", "").strip()
    if not raw_pairs:
        raw_pairs = DEFAULT_MODEL_CANDIDATE_PAIRS or f"{default_ner_model}:{default_classifier_model}"

    pairs: List[Dict[str, str]] = []
    for idx, item in enumerate(raw_pairs.split(","), start=1):
        candidate = item.strip()
        if not candidate:
            continue
        if ":" not in candidate:
            raise ValueError(
                "MODEL_CANDIDATE_PAIRS entries must use the format 'ner_model:classifier_model'"
            )
        ner_model, classifier_model = [part.strip() for part in candidate.split(":", 1)]
        if not ner_model or not classifier_model:
            raise ValueError(
                "MODEL_CANDIDATE_PAIRS entries must include both ner_model and classifier_model"
            )
        pairs.append({
            "label": f"candidate_{idx}_{ner_model}__{classifier_model}",
            "ner_model": ner_model,
            "classifier_model": classifier_model,
        })

    if not pairs:
        raise ValueError("No valid model candidate pairs were configured for retraining")
    return pairs


def locate_model_dir(prefix: str, model_name: str) -> Path:
    candidate = Path(f"/tmp/{prefix}-{model_name}")
    if not candidate.exists():
        raise FileNotFoundError(f"Expected trained model directory at {candidate}")
    return candidate


def package_candidate(
    ner_model_name: str,
    classifier_model_name: str,
    candidate_label: str | None = None,
) -> Dict[str, Any]:
    release_root = Path(os.getenv("MODEL_RELEASES_ROOT", DEFAULT_RELEASE_ROOT))
    release_root.mkdir(parents=True, exist_ok=True)

    release_label = (candidate_label or f"{ner_model_name}-{classifier_model_name}").replace("/", "-")
    candidate_version_base = os.getenv(
        "CANDIDATE_VERSION",
        f"candidate-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}",
    )
    candidate_version = f"{candidate_version_base}-{release_label}"
    candidate_root = release_root / candidate_version
    if candidate_root.exists():
        shutil.rmtree(candidate_root)
    candidate_root.mkdir(parents=True, exist_ok=True)

    ner_src = locate_model_dir("deadline-ner", ner_model_name)
    classifier_src = locate_model_dir("deadline-clf", classifier_model_name)
    ner_dst = candidate_root / "ner"
    classifier_dst = candidate_root / "classifier"
    shutil.copytree(ner_src, ner_dst)
    shutil.copytree(classifier_src, classifier_dst)

    onnx_export_root = candidate_root / "_onnx_export"
    quantized_clf_dst = candidate_root / "onnx_quantized_clf"
    quantized_ner_dst = candidate_root / "onnx_quantized_ner"
    quantize_env = os.environ.copy()
    quantize_env.update(
        {
            "ONNX_WORKDIR": str(onnx_export_root),
            "HF_CLF_SOURCE_PATH": str(classifier_dst),
            "HF_NER_SOURCE_PATH": str(ner_dst),
            "ONNX_CLF_EXPORT_PATH": str(onnx_export_root / "onnx_model_clf"),
            "ONNX_NER_EXPORT_PATH": str(onnx_export_root / "onnx_model_ner"),
            "ONNX_CLF_QUANTIZED_PATH": str(quantized_clf_dst),
            "ONNX_NER_QUANTIZED_PATH": str(quantized_ner_dst),
        }
    )
    serving_root = Path(os.getenv("SERVING_ROOT", "/app/components/serving"))
    quantize_command = ["python", str(serving_root / "quantize_onnx.py")]
    if run_command(quantize_command, env=quantize_env) != 0:
        raise RuntimeError("ONNX export/quantization failed for candidate package")
    shutil.rmtree(onnx_export_root, ignore_errors=True)

    package_manifest = {
        "candidate_version": candidate_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate_label": release_label,
        "ner_model_name": ner_model_name,
        "classifier_model_name": classifier_model_name,
        "ner_path": str(ner_dst),
        "classifier_path": str(classifier_dst),
        "onnx_classifier_path": str(quantized_clf_dst),
        "onnx_ner_path": str(quantized_ner_dst),
    }
    if object_store_enabled():
        bundle_key = f"releases/{candidate_version}/bundle.tar.gz"
        upload_directory_as_tarball(
            bucket=os.getenv("MODEL_ARTIFACT_BUCKET", DEFAULT_MODEL_ARTIFACT_BUCKET),
            key=bundle_key,
            source_dir=candidate_root,
            tmp_dir="/tmp",
        )
        package_manifest["bundle_s3_key"] = bundle_key
        upload_json(
            bucket=os.getenv("MODEL_ARTIFACT_BUCKET", DEFAULT_MODEL_ARTIFACT_BUCKET),
            key=f"releases/{candidate_version}/manifest.json",
            payload=package_manifest,
        )
    (candidate_root / "package_manifest.json").write_text(json.dumps(package_manifest, indent=2))
    return package_manifest


def load_model_run_metadata(model_dir: str) -> Dict[str, Any] | None:
    metadata_path = Path(model_dir) / "mlflow_run.json"
    if not metadata_path.exists():
        return None

    try:
        return json.loads(metadata_path.read_text())
    except json.JSONDecodeError:
        return None


def fetch_run_metrics(client: MlflowClient, run_id: str) -> Dict[str, Any]:
    run = client.get_run(run_id)
    return {
        "run_id": run.info.run_id,
        "status": run.info.status,
        "metrics": dict(run.data.metrics),
        "params": dict(run.data.params),
    }


def best_successful_experiment_metrics(
    tracking_uri: str,
    experiment_name: str,
    preferred_model_name: str | None = None,
) -> Dict[str, Any]:
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment not found: {experiment_name}")

    filters = ["attributes.status = 'FINISHED'"]
    if preferred_model_name:
        filters.append(f"params.model_name = '{preferred_model_name}'")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=" and ".join(filters),
        order_by=["metrics.test_f1 DESC", "attributes.start_time DESC"],
        max_results=25,
    )

    if not runs and preferred_model_name:
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="attributes.status = 'FINISHED'",
            order_by=["metrics.test_f1 DESC", "attributes.start_time DESC"],
            max_results=50,
        )
        runs = [
            run for run in runs
            if "baseline" not in dict(run.data.params).get("model_name", "").lower()
        ]

    if not runs:
        raise RuntimeError(f"No successful MLflow runs found for experiment: {experiment_name}")

    run = runs[0]
    return {
        "run_id": run.info.run_id,
        "status": run.info.status,
        "metrics": dict(run.data.metrics),
        "params": dict(run.data.params),
    }


def selected_experiment_metrics(
    tracking_uri: str,
    experiment_name: str,
    model_dir: str,
    preferred_model_name: str | None = None,
) -> Dict[str, Any]:
    metadata = load_model_run_metadata(model_dir)
    client = MlflowClient(tracking_uri=tracking_uri)

    if metadata and metadata.get("run_id"):
        run_metrics = fetch_run_metrics(client, str(metadata["run_id"]))
        if run_metrics["status"] == "FINISHED":
            return run_metrics
        raise RuntimeError(f"MLflow run {metadata['run_id']} for {model_dir} did not finish successfully")
    raise RuntimeError(
        f"Strict MLflow attribution failed for {model_dir}; missing usable mlflow_run.json metadata"
    )


def load_current_production_metrics() -> Dict[str, Any]:
    path = Path(os.getenv("CURRENT_PRODUCTION_METRICS_PATH", DEFAULT_CURRENT_PRODUCTION_METRICS_PATH))
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return {}
    if object_store_enabled():
        try:
            return download_json(
                os.getenv("RUNTIME_LOG_BUCKET", DEFAULT_RUNTIME_LOG_BUCKET),
                os.getenv("CURRENT_PRODUCTION_METRICS_S3_KEY", "automation/current_production_metrics.json"),
            )
        except Exception:
            return {}
    return {}


def load_feedback_checkpoint() -> Dict[str, Any]:
    checkpoint_path = Path(os.getenv("FEEDBACK_CHECKPOINT_PATH", DEFAULT_FEEDBACK_CHECKPOINT_PATH))
    if checkpoint_path.exists():
        try:
            return json.loads(checkpoint_path.read_text())
        except json.JSONDecodeError:
            return {}
    if object_store_enabled():
        try:
            return download_json(
                os.getenv("RUNTIME_LOG_BUCKET", DEFAULT_RUNTIME_LOG_BUCKET),
                os.getenv("FEEDBACK_CHECKPOINT_S3_KEY", "automation/feedback_checkpoint.json"),
            )
        except Exception:
            return {}
    return {}


def ensure_registered_model(client: MlflowClient, model_name: str) -> None:
    try:
        client.get_registered_model(model_name)
    except Exception:
        client.create_registered_model(model_name)


def register_candidate_models(
    tracking_uri: str,
    candidate_version: str,
    ner_run_id: str,
    classifier_run_id: str,
) -> Dict[str, Any]:
    client = MlflowClient(tracking_uri=tracking_uri)
    registry_map = {
        "ner": {
            "name": os.getenv("NER_REGISTERED_MODEL_NAME", DEFAULT_NER_REGISTERED_MODEL),
            "run_id": ner_run_id,
            "summary": "Candidate NER model from automated retraining cycle.",
        },
        "classifier": {
            "name": os.getenv("CLASSIFIER_REGISTERED_MODEL_NAME", DEFAULT_CLASSIFIER_REGISTERED_MODEL),
            "run_id": classifier_run_id,
            "summary": "Candidate classifier model from automated retraining cycle.",
        },
    }

    registered_versions: Dict[str, Any] = {}
    for component, spec in registry_map.items():
        model_name = spec["name"]
        ensure_registered_model(client, model_name)
        version = client.create_model_version(
            name=model_name,
            source=f"runs:/{spec['run_id']}/model",
            run_id=spec["run_id"],
            description=f"{spec['summary']} Candidate release: {candidate_version}",
        )
        client.set_model_version_tag(model_name, version.version, "candidate_version", candidate_version)
        client.set_model_version_tag(model_name, version.version, "release_channel", "candidate")
        client.set_model_version_tag(model_name, version.version, "component", component)
        try:
            client.set_registered_model_alias(model_name, "candidate", version.version)
        except Exception:
            pass
        registered_versions[component] = {
            "registered_model": model_name,
            "version": version.version,
            "run_id": spec["run_id"],
        }

    return registered_versions


def build_evaluation_metrics(package_manifest: Dict[str, Any]) -> Dict[str, Any]:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow.platform.svc.cluster.local:5000")
    training_root = Path(os.getenv("TRAINING_ROOT", "/app/components/training"))
    evaluation_path = Path(os.getenv("EVALUATION_METRICS_PATH", DEFAULT_EVAL_PATH))
    evaluation_path.parent.mkdir(parents=True, exist_ok=True)

    eval_env = os.environ.copy()
    eval_env["MLFLOW_TRACKING_URI"] = tracking_uri
    eval_env["MLFLOW_S3_ENDPOINT_URL"] = os.getenv(
        "MLFLOW_S3_ENDPOINT_URL", "http://minio.platform.svc.cluster.local:9000"
    )

    eval_command = [
        "python",
        str(training_root / "src" / "evaluate.py"),
        "--clf_model",
        package_manifest["classifier_path"],
        "--ner_model",
        package_manifest["ner_path"],
        "--threshold",
        os.getenv("EVAL_THRESHOLD", "0.7"),
        "--output_json",
        str(evaluation_path),
    ]

    cross_domain_samples = os.getenv("CROSS_DOMAIN_SAMPLES_PATH")
    if cross_domain_samples:
        eval_command.extend(["--cross_domain_samples", cross_domain_samples])

    if run_command(eval_command, env=eval_env) != 0:
        raise RuntimeError("End-to-end evaluation failed")

    evaluation_metrics = json.loads(evaluation_path.read_text())
    ner_metrics = selected_experiment_metrics(
        tracking_uri,
        os.getenv("NER_EXPERIMENT_NAME", "deadline-detection-ner"),
        package_manifest["ner_path"],
        package_manifest["ner_model_name"],
    )
    clf_metrics = selected_experiment_metrics(
        tracking_uri,
        os.getenv("CLASSIFIER_EXPERIMENT_NAME", "deadline-detection-classifier"),
        package_manifest["classifier_path"],
        package_manifest["classifier_model_name"],
    )

    evaluation_metrics.update(
        {
            "candidate_version": package_manifest["candidate_version"],
            "candidate_bundle_s3_key": package_manifest.get("bundle_s3_key"),
            "candidate_paths": {
                "ner": package_manifest["ner_path"],
                "classifier": package_manifest["classifier_path"],
                "onnx_classifier": package_manifest["onnx_classifier_path"],
                "onnx_ner": package_manifest["onnx_ner_path"],
            },
            "ner_f1": float(ner_metrics["metrics"].get("test_f1", 0.0)),
            "clf_macro_f1": float(clf_metrics["metrics"].get("test_f1", 0.0)),
            "mlflow_runs": {
                "ner": ner_metrics["run_id"],
                "classifier": clf_metrics["run_id"],
            },
        }
    )
    evaluation_path.write_text(json.dumps(evaluation_metrics, indent=2))
    maybe_upload_json("EVALUATION_METRICS_S3_KEY", "automation/evaluation_metrics.json", evaluation_metrics)
    return evaluation_metrics


def score_candidate(evaluation_metrics: Dict[str, Any]) -> float:
    total_contracts = max(int(evaluation_metrics.get("total_test_contracts", 0)), 1)
    false_alarm_rate = float(evaluation_metrics.get("false_alarm_count", 0)) / total_contracts
    coverage = float(evaluation_metrics.get("e2e_coverage", 0.0))
    exact_match = float(evaluation_metrics.get("exact_match_pct", 0.0)) / 100.0
    within_30 = float(evaluation_metrics.get("within_30_days_pct", 0.0)) / 100.0
    ner_f1 = float(evaluation_metrics.get("ner_f1", 0.0))
    clf_macro_f1 = float(evaluation_metrics.get("clf_macro_f1", 0.0))
    cross_domain_accuracy = float(evaluation_metrics.get("cross_domain_accuracy", 0.0))

    score = (
        (0.35 * coverage)
        + (0.15 * exact_match)
        + (0.10 * within_30)
        + (0.15 * ner_f1)
        + (0.15 * clf_macro_f1)
        + (0.10 * cross_domain_accuracy)
        - (0.10 * false_alarm_rate)
    )
    return round(score, 6)


def build_champion_comparison(
    evaluation_metrics: Dict[str, Any],
    champion_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    if not champion_metrics:
        return {"available": False}
    candidate_score = score_candidate(evaluation_metrics)
    champion_score = float(champion_metrics.get("candidate_score", champion_metrics.get("score", 0.0)))
    return {
        "available": True,
        "candidate_score": candidate_score,
        "champion_score": champion_score,
        "score_delta": round(candidate_score - champion_score, 6),
        "ner_f1_delta": round(
            float(evaluation_metrics.get("ner_f1", 0.0)) - float(champion_metrics.get("ner_f1", 0.0)),
            6,
        ),
        "clf_macro_f1_delta": round(
            float(evaluation_metrics.get("clf_macro_f1", 0.0)) - float(champion_metrics.get("clf_macro_f1", 0.0)),
            6,
        ),
        "coverage_delta": round(
            float(evaluation_metrics.get("e2e_coverage", 0.0)) - float(champion_metrics.get("e2e_coverage", 0.0)),
            6,
        ),
        "exact_match_delta": round(
            float(evaluation_metrics.get("exact_match_pct", 0.0)) - float(champion_metrics.get("exact_match_pct", 0.0)),
            6,
        ),
        "within_30_days_delta": round(
            float(evaluation_metrics.get("within_30_days_pct", 0.0)) - float(champion_metrics.get("within_30_days_pct", 0.0)),
            6,
        ),
    }


def training_quality_gate(
    evaluation_metrics: Dict[str, Any],
    champion_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    min_ner_f1 = float(os.getenv("MIN_NER_F1", "0.65"))
    min_clf_macro_f1 = float(os.getenv("MIN_CLF_MACRO_F1", "0.65"))
    min_e2e_coverage = float(os.getenv("MIN_E2E_COVERAGE", "0.60"))
    max_false_alarm_count = int(os.getenv("MAX_FALSE_ALARM_COUNT", "10"))
    min_candidate_score = float(os.getenv("MIN_CANDIDATE_SCORE", "0.45"))
    max_score_regression = float(os.getenv("MAX_SCORE_REGRESSION", "0.01"))
    max_ner_f1_regression = float(os.getenv("MAX_NER_F1_REGRESSION", "0.02"))
    max_clf_f1_regression = float(os.getenv("MAX_CLF_F1_REGRESSION", "0.02"))
    max_coverage_regression = float(os.getenv("MAX_E2E_COVERAGE_REGRESSION", "0.02"))

    failures: List[str] = []
    candidate_score = score_candidate(evaluation_metrics)
    if float(evaluation_metrics.get("ner_f1", 0.0)) < min_ner_f1:
        failures.append("ner_f1_below_threshold")
    if float(evaluation_metrics.get("clf_macro_f1", 0.0)) < min_clf_macro_f1:
        failures.append("clf_macro_f1_below_threshold")
    if float(evaluation_metrics.get("e2e_coverage", 0.0)) < min_e2e_coverage:
        failures.append("e2e_coverage_below_threshold")
    if int(evaluation_metrics.get("false_alarm_count", 999999)) > max_false_alarm_count:
        failures.append("false_alarm_count_above_threshold")
    if candidate_score < min_candidate_score:
        failures.append("candidate_score_below_threshold")

    comparison = build_champion_comparison(evaluation_metrics, champion_metrics)
    if comparison.get("available"):
        if comparison["score_delta"] < -max_score_regression:
            failures.append("candidate_score_regresses_vs_production")
        if comparison["ner_f1_delta"] < -max_ner_f1_regression:
            failures.append("ner_f1_regresses_vs_production")
        if comparison["clf_macro_f1_delta"] < -max_clf_f1_regression:
            failures.append("clf_macro_f1_regresses_vs_production")
        if comparison["coverage_delta"] < -max_coverage_regression:
            failures.append("e2e_coverage_regresses_vs_production")

    return {
        "eligible": not failures,
        "candidate_score": candidate_score,
        "failed_gates": failures,
        "comparison_to_current_production": comparison,
        "thresholds": {
            "min_ner_f1": min_ner_f1,
            "min_clf_macro_f1": min_clf_macro_f1,
            "min_e2e_coverage": min_e2e_coverage,
            "max_false_alarm_count": max_false_alarm_count,
            "min_candidate_score": min_candidate_score,
            "max_score_regression": max_score_regression,
            "max_ner_f1_regression": max_ner_f1_regression,
            "max_clf_f1_regression": max_clf_f1_regression,
            "max_coverage_regression": max_coverage_regression,
        },
    }


def orchestrate_training() -> Dict[str, Any]:
    training_root = Path(os.getenv("TRAINING_ROOT", "/app/components/training"))
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow.platform.svc.cluster.local:5000")
    training_data_path = Path(os.getenv("TRAINING_DATA_PATH", "/app/data/deadline_sentences"))
    shared_env = os.environ.copy()
    shared_env["MLFLOW_TRACKING_URI"] = tracking_uri
    shared_env["MLFLOW_S3_ENDPOINT_URL"] = os.getenv(
        "MLFLOW_S3_ENDPOINT_URL", "http://minio.platform.svc.cluster.local:9000"
    )

    default_ner_model = os.getenv("NER_MODEL_NAME", "bert_ner_v4")
    default_classifier_model = os.getenv("CLASSIFIER_MODEL_NAME", "roberta_clf_v3")
    candidate_pairs = parse_model_candidate_pairs(default_ner_model, default_classifier_model)
    config = {
        "default_ner_model": default_ner_model,
        "default_classifier_model": default_classifier_model,
        "candidate_pairs": candidate_pairs,
    }

    commands: List[List[str]] = []
    if os.getenv("FORCE_REBUILD_DATASET", "false").lower() in {"1", "true", "yes"} or not training_data_path.exists():
        commands.append(
            [
                "python",
                str(training_root / "src" / "build_dataset.py"),
                "--save_path",
                str(training_data_path),
            ]
        )

    results = []
    for command in commands:
        exit_code = run_command(command, env=shared_env)
        results.append({"command": command, "exit_code": exit_code})
        if exit_code != 0:
            return {"ok": False, "steps": results, "config": config}

    feedback_checkpoint = load_feedback_checkpoint()
    feedback_curation_dir = Path(os.getenv("FEEDBACK_CURATION_DIR", "/tmp/feedback-curation"))
    feedback_curation_summary = compile_feedback_training_additions(
        output_dir=str(feedback_curation_dir),
        checkpoint=feedback_checkpoint,
    )
    champion_metrics = load_current_production_metrics()
    evaluated_candidates: List[Dict[str, Any]] = []
    for candidate in candidate_pairs:
        candidate_steps: List[Dict[str, Any]] = []
        candidate_env = shared_env.copy()
        candidate_env["FEEDBACK_CLASSIFIER_ADDITIONS_PATH"] = feedback_curation_summary["classifier_additions_path"]
        candidate_env["FEEDBACK_NER_ADDITIONS_PATH"] = feedback_curation_summary["ner_additions_path"]

        if os.getenv("ENABLE_RAY_TUNE_FOR_RETRAIN", "false").lower() in {"1", "true", "yes"}:
            tuning_dir = Path(os.getenv("RAY_TUNE_CONFIG_DIR", "/tmp/ray-tune-configs"))
            tuning_dir.mkdir(parents=True, exist_ok=True)
            ner_tune_path = tuning_dir / f"{candidate['label']}_ner.json"
            clf_tune_path = tuning_dir / f"{candidate['label']}_clf.json"
            ner_tune_env = candidate_env.copy()
            clf_tune_env = candidate_env.copy()
            ner_tune_env["BEST_CONFIG_OUT"] = str(ner_tune_path)
            clf_tune_env["BEST_CONFIG_OUT"] = str(clf_tune_path)
            if run_command(["python", str(training_root / "src" / "train_ner_ray_tune.py")], env=ner_tune_env) != 0:
                return {"ok": False, "steps": results, "config": config, "error": "NER Ray Tune failed"}
            if run_command(["python", str(training_root / "src" / "train_classifier_ray_tune.py")], env=clf_tune_env) != 0:
                return {"ok": False, "steps": results, "config": config, "error": "Classifier Ray Tune failed"}
            ner_best = json.loads(ner_tune_path.read_text()) if ner_tune_path.exists() else {}
            clf_best = json.loads(clf_tune_path.read_text()) if clf_tune_path.exists() else {}
            candidate["ner_config_json"] = json.dumps(ner_best.get("config", {}))
            candidate["classifier_config_json"] = json.dumps(clf_best.get("config", {}))

        train_commands = [
            [
                "python", str(training_root / "src" / "train_ner.py"),
                "--model", candidate["ner_model"],
                "--config_json", candidate.get("ner_config_json", ""),
            ],
            [
                "python", str(training_root / "src" / "train_classifier.py"),
                "--model", candidate["classifier_model"],
                "--config_json", candidate.get("classifier_config_json", ""),
            ],
        ]

        candidate_failed = False
        for command in train_commands:
            exit_code = run_command(command, env=candidate_env)
            step_result = {
                "command": command,
                "exit_code": exit_code,
                "candidate_label": candidate["label"],
            }
            candidate_steps.append(step_result)
            results.append(step_result)
            if exit_code != 0:
                candidate_failed = True
                break

        if candidate_failed:
            evaluated_candidates.append({
                **candidate,
                "ok": False,
                "steps": candidate_steps,
            })
            continue

        try:
            package_manifest = package_candidate(
                candidate["ner_model"],
                candidate["classifier_model"],
                candidate["label"],
            )
            evaluation_metrics = build_evaluation_metrics(package_manifest)
            quality_gate = training_quality_gate(evaluation_metrics, champion_metrics)
            evaluation_metrics["candidate_score"] = quality_gate["candidate_score"]
            evaluation_metrics["registration_eligible"] = quality_gate["eligible"]
            evaluation_metrics["training_quality_gate"] = quality_gate
            if quality_gate["eligible"]:
                evaluation_metrics["model_registry"] = register_candidate_models(
                    tracking_uri=tracking_uri,
                    candidate_version=package_manifest["candidate_version"],
                    ner_run_id=evaluation_metrics["mlflow_runs"]["ner"],
                    classifier_run_id=evaluation_metrics["mlflow_runs"]["classifier"],
                )
            else:
                evaluation_metrics["model_registry"] = {}
            evaluation_path = Path(os.getenv("EVALUATION_METRICS_PATH", DEFAULT_EVAL_PATH))
            evaluation_path.write_text(json.dumps(evaluation_metrics, indent=2))
            maybe_upload_json("EVALUATION_METRICS_S3_KEY", "automation/evaluation_metrics.json", evaluation_metrics)
            candidate_score = score_candidate(evaluation_metrics)
            evaluated_candidates.append({
                **candidate,
                "ok": True,
                "steps": candidate_steps,
                "package_manifest": package_manifest,
                "evaluation_metrics": evaluation_metrics,
                "score": candidate_score,
            })
        except Exception as exc:
            evaluated_candidates.append({
                **candidate,
                "ok": False,
                "steps": candidate_steps,
                "error": str(exc),
            })

    successful_candidates = [candidate for candidate in evaluated_candidates if candidate.get("ok")]
    eligible_candidates = [
        candidate
        for candidate in successful_candidates
        if candidate.get("evaluation_metrics", {}).get("registration_eligible")
    ]
    if not successful_candidates:
        return {
            "ok": False,
            "steps": results,
            "config": config,
            "feedback_curation": feedback_curation_summary,
            "evaluated_candidates": evaluated_candidates,
        }
    if not eligible_candidates:
        return {
            "ok": False,
            "steps": results,
            "config": config,
            "feedback_curation": feedback_curation_summary,
            "evaluated_candidates": evaluated_candidates,
            "error": "no_candidates_passed_training_quality_gate",
        }

    best_candidate = max(eligible_candidates, key=lambda candidate: candidate.get("score", float("-inf")))
    return {
        "ok": True,
        "steps": results,
        "config": config,
        "feedback_curation": feedback_curation_summary,
        "evaluated_candidates": evaluated_candidates,
        "selected_candidate": {
            "label": best_candidate["label"],
            "ner_model": best_candidate["ner_model"],
            "classifier_model": best_candidate["classifier_model"],
            "score": best_candidate["score"],
        },
        "package_manifest": best_candidate["package_manifest"],
        "evaluation_metrics": best_candidate["evaluation_metrics"],
    }


def main() -> int:
    decision = should_retrain(load_feedback_metrics())
    output_path = Path(os.getenv("RETRAIN_DECISION_PATH", DEFAULT_OUTPUT_PATH))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if os.getenv("DRY_RUN", "false").lower() == "true":
        output_path.write_text(json.dumps(decision, indent=2))
        maybe_upload_json("RETRAIN_DECISION_S3_KEY", "automation/retrain_decision.json", decision)
        print(json.dumps(decision, indent=2))
        return 0

    if not decision["trigger_retrain"]:
        decision["status"] = "skipped"
        output_path.write_text(json.dumps(decision, indent=2))
        maybe_upload_json("RETRAIN_DECISION_S3_KEY", "automation/retrain_decision.json", decision)
        print(json.dumps(decision, indent=2))
        return 0

    training_result = orchestrate_training()
    decision["status"] = "completed" if training_result["ok"] else "failed"
    decision["training"] = training_result
    if training_result.get("package_manifest"):
        decision["candidate_version"] = training_result["package_manifest"]["candidate_version"]
        decision["candidate_paths"] = {
            "ner": training_result["package_manifest"]["ner_path"],
            "classifier": training_result["package_manifest"]["classifier_path"],
            "onnx_classifier": training_result["package_manifest"]["onnx_classifier_path"],
            "onnx_ner": training_result["package_manifest"]["onnx_ner_path"],
        }
        if training_result["package_manifest"].get("bundle_s3_key"):
            decision["candidate_bundle_s3_key"] = training_result["package_manifest"]["bundle_s3_key"]

    if training_result["ok"]:
        # Advance the checkpoint so the next scheduled run only counts new events.
        feedback_log_path = Path(os.getenv("FEEDBACK_LOG_PATH", "/tmp/feedback_events.jsonl"))
        checkpoint_path = Path(os.getenv("FEEDBACK_CHECKPOINT_PATH", DEFAULT_FEEDBACK_CHECKPOINT_PATH))
        total_lines = 0
        if object_store_enabled():
            total_lines = len(
                load_json_objects(
                    os.getenv("RUNTIME_LOG_BUCKET", DEFAULT_RUNTIME_LOG_BUCKET),
                    os.getenv("FEEDBACK_S3_PREFIX", "runtime/online-features/feedback"),
                )
            )
        elif feedback_log_path.exists():
            total_lines = sum(1 for line in feedback_log_path.read_text().splitlines() if line.strip())
        if total_lines:
            latest_feedback_timestamp = (
                training_result
                .get("feedback_curation", {})
                .get("latest_new_feedback_timestamp")
                or training_result.get("feedback_curation", {}).get("latest_used_feedback_timestamp")
                or datetime.now(timezone.utc).isoformat()
            )
            checkpoint_payload = {
                "last_trained_object_count": total_lines,
                "last_trained_at": datetime.now(timezone.utc).isoformat(),
                "last_trained_feedback_timestamp": latest_feedback_timestamp,
                "candidate_version": decision.get("candidate_version"),
            }
            checkpoint_path.write_text(json.dumps({
                **checkpoint_payload,
                "last_trained_line_count": total_lines,
            }, indent=2))
            maybe_upload_json("FEEDBACK_CHECKPOINT_S3_KEY", "automation/feedback_checkpoint.json", checkpoint_payload)

    output_path.write_text(json.dumps(decision, indent=2))
    maybe_upload_json("RETRAIN_DECISION_S3_KEY", "automation/retrain_decision.json", decision)
    print(json.dumps(decision, indent=2))
    return 0 if training_result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
