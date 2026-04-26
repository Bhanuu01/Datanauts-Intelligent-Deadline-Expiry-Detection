import json
import os
import shutil
import sys
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List

DEFAULT_PROMOTION_PATH = "/tmp/promotion_decision.json"
DEFAULT_RELEASE_PLAN_PATH = "/tmp/release_plan.json"
DEFAULT_RELEASE_STATE_PATH = "/tmp/release_state.json"
DEFAULT_RELEASES_ROOT = "/data/model-releases"


def load_json(path_env: str, default_path: str) -> Dict[str, Any]:
    path = Path(os.getenv(path_env, default_path))
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def load_release_state() -> Dict[str, Any]:
    state_path = Path(os.getenv("RELEASE_STATE_PATH", DEFAULT_RELEASE_STATE_PATH))
    if not state_path.exists():
        return {
            "current_version": "bootstrap",
            "previous_version": None,
            "current_stage": os.getenv("CURRENT_RELEASE_STAGE", "staging"),
        }
    return json.loads(state_path.read_text())


def save_release_state(state: Dict[str, Any]) -> None:
    state_path = Path(os.getenv("RELEASE_STATE_PATH", DEFAULT_RELEASE_STATE_PATH))
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2))


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
        "candidate_paths": promotion_decision.get("candidate_paths", {}),
        "candidate_quantized_paths": promotion_decision.get("candidate_quantized_paths", {}),
        "current_version": release_state.get("current_version"),
        "previous_version": release_state.get("previous_version"),
        "metrics": {
            "ner_f1": promotion_decision.get("ner_f1"),
            "clf_macro_f1": promotion_decision.get("clf_macro_f1"),
            "e2e_coverage": promotion_decision.get("e2e_coverage"),
            "false_alarm_count": promotion_decision.get("false_alarm_count"),
            "candidate_latency_p95_ms": promotion_decision.get("candidate_latency_p95_ms"),
        },
    }


def replace_directory(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def canonicalize_current_release(release_state: Dict[str, Any], canonical_root: Path, releases_root: Path) -> None:
    current_version = release_state.get("current_version")
    if not current_version or current_version == "bootstrap":
        return

    current_release_root = releases_root / current_version
    ner_release_root = current_release_root / "ner"
    clf_release_root = current_release_root / "classifier"
    quantized_release_root = current_release_root / "onnx_quantized_model"
    quantized_clf_release_root = quantized_release_root / "onnx_quantized_clf"
    quantized_ner_release_root = quantized_release_root / "onnx_quantized_ner"
    if (
        ner_release_root.exists()
        and clf_release_root.exists()
        and quantized_clf_release_root.exists()
        and quantized_ner_release_root.exists()
    ):
        return

    current_release_root.mkdir(parents=True, exist_ok=True)
    if (canonical_root / "ner").exists():
        shutil.copytree(canonical_root / "ner", ner_release_root, dirs_exist_ok=True)
    if (canonical_root / "classifier").exists():
        shutil.copytree(canonical_root / "classifier", clf_release_root, dirs_exist_ok=True)
    if (canonical_root / "onnx_quantized_model" / "onnx_quantized_clf").exists():
        shutil.copytree(
            canonical_root / "onnx_quantized_model" / "onnx_quantized_clf",
            quantized_clf_release_root,
            dirs_exist_ok=True,
        )
    if (canonical_root / "onnx_quantized_model" / "onnx_quantized_ner").exists():
        shutil.copytree(
            canonical_root / "onnx_quantized_model" / "onnx_quantized_ner",
            quantized_ner_release_root,
            dirs_exist_ok=True,
        )


def patch_live_service(namespace: str, service_name: str, selector: Dict[str, str]) -> None:
    from kubernetes import client

    api = client.CoreV1Api()
    body = {"spec": {"selector": selector}}
    api.patch_namespaced_service(name=service_name, namespace=namespace, body=body)


def restart_release_deployments(namespace: str, deployment_names: List[str]) -> None:
    from kubernetes import client

    apps = client.AppsV1Api()
    restart_annotation = datetime.now(timezone.utc).isoformat()
    for deployment_name in deployment_names:
        body = {
            "spec": {
                "template": {
                    "metadata": {
                        "annotations": {
                            "kubectl.kubernetes.io/restartedAt": restart_annotation
                        }
                    }
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
    canonical_root = Path(os.getenv("MODEL_CANONICAL_ROOT", "/data"))
    releases_root = Path(os.getenv("MODEL_RELEASES_ROOT", DEFAULT_RELEASES_ROOT))
    releases_root.mkdir(parents=True, exist_ok=True)

    skip_k8s_apply = os.getenv("SKIP_K8S_APPLY", "false").lower() in {"1", "true", "yes"}
    if not skip_k8s_apply:
        try:
            from kubernetes import config

            config.load_incluster_config()
        except Exception:
            from kubernetes import config

            config.load_kube_config()

    canonicalize_current_release(release_state, canonical_root, releases_root)

    if release_plan["action"] == "promote":
        candidate_paths = release_plan.get("candidate_paths", {})
        candidate_quantized_paths = release_plan.get("candidate_quantized_paths", {})
        ner_source = Path(candidate_paths.get("ner", ""))
        clf_source = Path(candidate_paths.get("classifier", ""))
        quantized_ner_source = Path(candidate_quantized_paths.get("ner", ""))
        quantized_clf_source = Path(candidate_quantized_paths.get("classifier", ""))
        if not ner_source.exists() or not clf_source.exists():
            raise FileNotFoundError("Candidate model paths are missing; cannot promote release")
        if not quantized_ner_source.exists() or not quantized_clf_source.exists():
            raise FileNotFoundError("Candidate ONNX model paths are missing; cannot promote release")

        previous_version = release_state.get("current_version")
        replace_directory(ner_source, canonical_root / "ner")
        replace_directory(clf_source, canonical_root / "classifier")
        onnx_root = canonical_root / "onnx_quantized_model"
        onnx_root.mkdir(parents=True, exist_ok=True)
        replace_directory(quantized_clf_source, onnx_root / "onnx_quantized_clf")
        replace_directory(quantized_ner_source, onnx_root / "onnx_quantized_ner")
        release_state["previous_version"] = previous_version
        release_state["current_version"] = release_plan["candidate_version"]
        release_state["current_stage"] = release_plan["next_stage"]
        release_state["last_action"] = "promote"
    else:
        previous_version = release_state.get("previous_version")
        if not previous_version:
            raise RuntimeError("Rollback requested but no previous version is available")

        rollback_root = releases_root / previous_version
        replace_directory(rollback_root / "ner", canonical_root / "ner")
        replace_directory(rollback_root / "classifier", canonical_root / "classifier")
        replace_directory(
            rollback_root / "onnx_quantized_model" / "onnx_quantized_clf",
            canonical_root / "onnx_quantized_model" / "onnx_quantized_clf",
        )
        replace_directory(
            rollback_root / "onnx_quantized_model" / "onnx_quantized_ner",
            canonical_root / "onnx_quantized_model" / "onnx_quantized_ner",
        )
        release_state["current_version"] = previous_version
        release_state["current_stage"] = release_plan["next_stage"]
        release_state["last_action"] = "rollback"

    if not skip_k8s_apply:
        patch_live_service(namespace, release_plan["target_service"], release_plan["target_selector"])
        restart_release_deployments(
            namespace,
            [
                "deadline-onnx-serving",
                "deadline-onnx-serving-staging",
                "deadline-onnx-serving-canary",
                "deadline-onnx-serving-production",
            ],
        )
    save_release_state(release_state)
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
    print(json.dumps(release_plan, indent=2))
    return 0 if release_plan["action"] != "reject" else 1


if __name__ == "__main__":
    sys.exit(main())
