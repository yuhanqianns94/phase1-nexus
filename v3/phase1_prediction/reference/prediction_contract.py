"""Current prediction identity and validation of saved prediction inputs."""

from collections.abc import Mapping
from typing import Any


PREDICTION_MODEL_VERSION = "phase1_v3_no_uplift_v1"


def ensure_current_prediction(payload: Mapping[str, Any]) -> None:
    """Reject saved adjusted curves before recomputing probabilities or OC.

    This only recognizes obsolete metadata; it cannot reconstruct or apply the
    removed adjustment. Unadjusted historical results remain readable.
    """
    pending = [payload]
    seen: set[int] = set()
    while pending:
        node = pending.pop()
        if id(node) in seen:
            continue
        seen.add(id(node))
        for key, value in node.items():
            if not isinstance(value, Mapping):
                continue
            if key in {"target_trial_type_prior_adjustment", "combination_adjustment"}:
                if value.get("applied") or (
                    value.get("enabled") and value.get("trial_type") != "single"
                ):
                    raise ValueError(
                        "This saved prediction used the removed combination uplift. "
                        "Regenerate the prediction before querying doses or running BOIN."
                    )
            pending.append(value)
