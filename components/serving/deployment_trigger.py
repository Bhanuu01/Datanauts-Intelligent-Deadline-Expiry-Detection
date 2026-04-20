import json
import os
from pathlib import Path
from typing import Any
from typing import Dict


DEFAULT_PROMOTION_PATH = "/data/promotion_decision.json"


def load_promotion_decision() -> Dict[str, Any]:
    path = Path(os.getenv("PROMOTION_DECISION_PATH", DEFAULT_PROMOTION_PATH))
    if not path.exists():
        return {"promote": False, "reason": "promotion_decision_missing"}
    return json.loads(path.read_text())


def main() -> None:
    decision = load_promotion_decision()
    if decision.get("promote"):
        print("Optimized ONNX deployment can be promoted as canary or alternate serving path.")
    else:
        print(f"Optimized ONNX deployment not promoted: {decision.get('reason', 'unknown')}")


if __name__ == "__main__":
    main()
