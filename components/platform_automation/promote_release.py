import json
import os
import sys
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List


DEFAULT_PROMOTION_PATH = "/tmp/promotion_decision.json"
DEFAULT_RELEASE_PLAN_PATH = "/tmp/release_plan.json"


def load_json(path_env: str, default_path: str) -> Dict[str, Any]:
    path = Path(os.getenv(path_env, default_path))
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def build_release_plan(promotion_decision: Dict[str, Any]) -> Dict[str, Any]:
    current_stage = os.getenv("CURRENT_RELEASE_STAGE", "staging")
    release_order: List[str] = ["staging", "canary", "production"]
    if current_stage not in release_order:
        current_stage = "staging"

    promote = bool(promotion_decision.get("promote", False))
    failed_gates = promotion_decision.get("failed_gates", [])

    next_stage = current_stage
    if promote:
        current_index = release_order.index(current_stage)
        next_stage = release_order[min(current_index + 1, len(release_order) - 1)]

    return {
        "current_stage": current_stage,
        "promote": promote,
        "next_stage": next_stage,
        "action": "promote" if promote and next_stage != current_stage else ("hold" if promote else "reject"),
        "failed_gates": failed_gates,
        "promotion_reason": promotion_decision.get("reason", "promotion_decision_missing"),
        "metrics": {
            "ner_f1": promotion_decision.get("ner_f1"),
            "clf_macro_f1": promotion_decision.get("clf_macro_f1"),
            "e2e_coverage": promotion_decision.get("e2e_coverage"),
            "false_alarm_count": promotion_decision.get("false_alarm_count"),
            "candidate_latency_p95_ms": promotion_decision.get("candidate_latency_p95_ms"),
        },
    }


def main() -> int:
    promotion_decision = load_json("PROMOTION_DECISION_PATH", DEFAULT_PROMOTION_PATH)
    release_plan = build_release_plan(promotion_decision)

    output_path = Path(os.getenv("RELEASE_PLAN_PATH", DEFAULT_RELEASE_PLAN_PATH))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(release_plan, indent=2))
    print(json.dumps(release_plan, indent=2))
    return 0 if release_plan["action"] != "reject" else 1


if __name__ == "__main__":
    sys.exit(main())
