import json
import os
import sys
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List

from mlflow.tracking import MlflowClient
from components.common.object_store import download_json
from components.common.object_store import object_store_enabled
from components.common.object_store import upload_json

DEFAULT_PROMOTION_PATH = "/tmp/promotion_decision.json"
DEFAULT_RELEASE_PLAN_PATH = "/tmp/release_plan.json"
DEFAULT_RELEASE_STATE_PATH = "/tmp/release_state.json"
DEFAULT_CURRENT_PRODUCTION_METRICS_PATH = "/tmp/current_production_metrics.json"
DEFAULT_MLFLOW_TRACKING_URI = "http://mlflow.platform.svc.cluster.local:5000"


def load_json(path_env: str, default_path: str) -> Dict[str, Any]:
    path = Path(os.getenv(path_env, default_path))
    if path.exists():
        return json.loads(path.read_text())
    if object_store_enabled():
        try:
            return download_json(
                os.getenv("RUNTIME_LOG_BUCKET", "datanauts-runtime"),
                os.getenv(f"{path_env}_S3_KEY", default_path.replace("/tmp/", "automation/")),
            )
        except Exception:
            return {}
    return {}


def load_release_state() -> Dict[str, Any]:
    state_path = Path(os.getenv("RELEASE_STATE_PATH", DEFAULT_RELEASE_STATE_PATH))
    bootstrap_bundle_key = os.getenv("BOOTSTRAP_MODEL_BUNDLE_S3_KEY", "releases/bootstrap/bundle.tar.gz")
    default_state = {
        "current_version": "bootstrap",
        "previous_version": None,
        "current_stage": os.getenv("CURRENT_RELEASE_STAGE", "staging"),
        "current_bundle_s3_key": bootstrap_bundle_key,
        "previous_bundle_s3_key": None,
    }
    if state_path.exists():
        return json.loads(state_path.read_text())
    if object_store_enabled():
        try:
            return download_json(
                os.getenv("RUNTIME_LOG_BUCKET", "datanauts-runtime"),
                os.getenv("RELEASE_STATE_S3_KEY", "automation/release_state.json"),
            )
        except Exception:
            return default_state
    return default_state


def save_release_state(state: Dict[str, Any]) -> None:
    state_path = Path(os.getenv("RELEASE_STATE_PATH", DEFAULT_RELEASE_STATE_PATH))
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2))
    if object_store_enabled():
        upload_json(
            os.getenv("RUNTIME_LOG_BUCKET", "datanauts-runtime"),
            os.getenv("RELEASE_STATE_S3_KEY", "automation/release_state.json"),
            state,
        )


def save_current_production_metrics(metrics: Dict[str, Any]) -> None:
    metrics_path = Path(os.getenv("CURRENT_PRODUCTION_METRICS_PATH", DEFAULT_CURRENT_PRODUCTION_METRICS_PATH))
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    if object_store_enabled():
        upload_json(
            os.getenv("RUNTIME_LOG_BUCKET", "datanauts-runtime"),
            os.getenv("CURRENT_PRODUCTION_METRICS_S3_KEY", "automation/current_production_metrics.json"),
            metrics,
        )


def set_model_aliases(model_registry: Dict[str, Any], stage_alias: str) -> None:
    if os.getenv("SKIP_MODEL_ALIAS_UPDATES", "false").lower() in {"1", "true", "yes"}:
        return
    if not model_registry:
        return
    client = MlflowClient(tracking_uri=os.getenv("MLFLOW_TRACKING_URI", DEFAULT_MLFLOW_TRACKING_URI))
    for component, info in model_registry.items():
        model_name = info.get("registered_model")
        version = info.get("version")
        if not model_name or version is None:
            continue
        version_str = str(version)
        client.set_registered_model_alias(model_name, stage_alias, version_str)
        client.set_registered_model_alias(model_name, f"{component}-{stage_alias}", version_str)
        if stage_alias == "production":
            client.set_registered_model_alias(model_name, "champion", version_str)


def build_release_plan(promotion_decision: Dict[str, Any], release_state: Dict[str, Any]) -> Dict[str, Any]:
    current_stage = release_state.get("current_stage", os.getenv("CURRENT_RELEASE_STAGE", "staging"))
    release_order: List[str] = ["staging", "canary", "production"]
    if current_stage not in release_order:
        current_stage = "staging"

    promote = bool(promotion_decision.get("promote", False))
    rollback = bool(promotion_decision.get("rollback", False))
    failed_gates = promotion_decision.get("failed_gates", [])
    rollback_reasons = promotion_decision.get("rollback_reasons", [])

    if rollback:
        current_index = release_order.index(current_stage)
        next_stage = release_order[max(current_index - 1, 0)]
        action = "rollback" if next_stage != current_stage or release_state.get("previous_version") else "hold"
    else:
        next_stage = current_stage
        if promote:
            current_index = release_order.index(current_stage)
            next_stage = release_order[min(current_index + 1, len(release_order) - 1)]
        action = "promote" if promote else "reject"

    stage_to_app = {
        "staging": "deadline-onnx-serving-staging",
        "canary": "deadline-onnx-serving-canary",
        "production": "deadline-onnx-serving-production",
    }

    return {
        "current_stage": current_stage,
        "promote": promote,
        "rollback": rollback,
        "next_stage": next_stage,
        "action": action,
        "failed_gates": failed_gates,
        "rollback_reasons": rollback_reasons,
        "promotion_reason": promotion_decision.get("reason", "promotion_decision_missing"),
        "target_service": os.getenv("LIVE_SERVICE_NAME", "deadline-onnx-serving"),
        "target_selector": {"app": stage_to_app[next_stage]},
        "candidate_version": promotion_decision.get("candidate_version"),
        "candidate_bundle_s3_key": promotion_decision.get("candidate_bundle_s3_key"),
        "candidate_paths": promotion_decision.get("candidate_paths", {}),
        "model_registry": promotion_decision.get("model_registry", {}),
        "current_version": release_state.get("current_version"),
        "previous_version": release_state.get("previous_version"),
        "current_model_registry": release_state.get("current_model_registry", {}),
        "previous_model_registry": release_state.get("previous_model_registry", {}),
        "metrics": {
            "ner_f1": promotion_decision.get("ner_f1"),
            "clf_macro_f1": promotion_decision.get("clf_macro_f1"),
            "e2e_coverage": promotion_decision.get("e2e_coverage"),
            "false_alarm_count": promotion_decision.get("false_alarm_count"),
            "candidate_latency_p95_ms": promotion_decision.get("candidate_latency_p95_ms"),
            "candidate_score": promotion_decision.get("candidate_score"),
            "exact_match_pct": promotion_decision.get("exact_match_pct"),
            "within_30_days_pct": promotion_decision.get("within_30_days_pct"),
            "cross_domain_accuracy": promotion_decision.get("cross_domain_accuracy"),
        },
    }


def patch_live_service(namespace: str, service_name: str, selector: Dict[str, str]) -> None:
    from kubernetes import client

    api = client.CoreV1Api()
    body = {"spec": {"selector": selector}}
    api.patch_namespaced_service(name=service_name, namespace=namespace, body=body)


def patch_release_deployment_bundle(namespace: str, deployment_name: str, bundle_key: str) -> None:
    from kubernetes import client

    apps = client.AppsV1Api()
    restart_annotation = datetime.now(timezone.utc).isoformat()
    body = {
        "spec": {
            "template": {
                "metadata": {
                    "annotations": {
                        "kubectl.kubernetes.io/restartedAt": restart_annotation
                    }
                },
                "spec": {
                    "containers": [
                        {
                            "name": deployment_name,
                            "env": [
                                {"name": "MODEL_BUNDLE_S3_KEY", "value": bundle_key},
                                {"name": "MODEL_ARTIFACT_BUCKET", "value": os.getenv("MODEL_ARTIFACT_BUCKET", "datanauts-models")},
                                {"name": "MODEL_LOCAL_CACHE_ROOT", "value": "/tmp/model-cache"},
                            ],
                        }
                    ]
                },
            }
        }
    }
    apps.patch_namespaced_deployment(name=deployment_name, namespace=namespace, body=body)


def apply_release_plan(release_plan: Dict[str, Any], release_state: Dict[str, Any]) -> Dict[str, Any]:
    release_plan["applied"] = False
    if os.getenv("AUTO_APPLY_RELEASE", "false").lower() not in {"1", "true", "yes"}:
        return release_plan

    if release_plan["action"] not in {"promote", "rollback"}:
        return release_plan

    namespace = os.getenv("K8S_NAMESPACE", "ml")

    skip_k8s_apply = os.getenv("SKIP_K8S_APPLY", "false").lower() in {"1", "true", "yes"}
    if not skip_k8s_apply:
        try:
            from kubernetes import config

            config.load_incluster_config()
        except Exception:
            from kubernetes import config

            config.load_kube_config()

    stage_to_deployment = {
        "staging": "deadline-onnx-serving-staging",
        "canary": "deadline-onnx-serving-canary",
        "production": "deadline-onnx-serving-production",
    }

    if release_plan["action"] == "promote":
        previous_version = release_state.get("current_version")
        previous_bundle_s3_key = release_state.get("current_bundle_s3_key")
        previous_metrics = release_state.get("current_metrics")
        previous_model_registry = release_state.get("current_model_registry")
        candidate_bundle_s3_key = release_plan.get("candidate_bundle_s3_key")
        if not candidate_bundle_s3_key:
            raise FileNotFoundError("Candidate bundle key is missing; cannot promote release")
        release_state["previous_version"] = previous_version
        release_state["previous_bundle_s3_key"] = previous_bundle_s3_key
        release_state["previous_metrics"] = previous_metrics
        release_state["previous_model_registry"] = previous_model_registry
        release_state["current_version"] = release_plan["candidate_version"]
        release_state["current_bundle_s3_key"] = candidate_bundle_s3_key
        release_state["current_stage"] = release_plan["next_stage"]
        release_state["current_metrics"] = release_plan.get("metrics", {})
        release_state["current_model_registry"] = release_plan.get("model_registry", {})
        release_state["last_action"] = "promote"
    else:
        previous_version = release_state.get("previous_version")
        previous_bundle_s3_key = release_state.get("previous_bundle_s3_key")
        previous_metrics = release_state.get("previous_metrics", {})
        previous_model_registry = release_state.get("previous_model_registry", {})
        if not previous_version:
            raise RuntimeError("Rollback requested but no previous version is available")
        if not previous_bundle_s3_key:
            raise RuntimeError("Rollback requested but no previous bundle key is available")
        release_state["current_version"] = previous_version
        release_state["current_bundle_s3_key"] = previous_bundle_s3_key
        release_state["current_metrics"] = previous_metrics
        release_state["current_model_registry"] = previous_model_registry
        release_state["previous_version"] = None
        release_state["previous_bundle_s3_key"] = None
        release_state["previous_metrics"] = None
        release_state["previous_model_registry"] = None
        release_state["current_stage"] = release_plan["next_stage"]
        release_state["last_action"] = "rollback"

    if not skip_k8s_apply:
        patch_live_service(namespace, release_plan["target_service"], release_plan["target_selector"])
        deployment_name = stage_to_deployment[release_plan["next_stage"]]
        bundle_key = release_state.get("current_bundle_s3_key", release_plan.get("candidate_bundle_s3_key", ""))
        if bundle_key:
            patch_release_deployment_bundle(namespace, deployment_name, bundle_key)
    save_release_state(release_state)
    if release_state.get("current_metrics"):
        save_current_production_metrics(release_state["current_metrics"])
    if release_state.get("current_model_registry"):
        set_model_aliases(release_state["current_model_registry"], release_state["current_stage"])
    release_plan["applied"] = True
    release_plan["release_state"] = release_state
    return release_plan


def main() -> int:
    promotion_decision = load_json("PROMOTION_DECISION_PATH", DEFAULT_PROMOTION_PATH)
    release_state = load_release_state()
    release_plan = build_release_plan(promotion_decision, release_state)
    release_plan = apply_release_plan(release_plan, release_state)

    output_path = Path(os.getenv("RELEASE_PLAN_PATH", DEFAULT_RELEASE_PLAN_PATH))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(release_plan, indent=2))
    if object_store_enabled():
        upload_json(
            os.getenv("RUNTIME_LOG_BUCKET", "datanauts-runtime"),
            os.getenv("RELEASE_PLAN_S3_KEY", "automation/release_plan.json"),
            release_plan,
        )
    print(json.dumps(release_plan, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
