"""Current V3 Bayesian API. Archived baseline lives in legacy_service."""
from .multidrug.service import predict as predict_target_curve
from .multidrug.service import preview as preview_target_evidence
from .multidrug import VERSION as PREDICTION_MODEL_VERSION
