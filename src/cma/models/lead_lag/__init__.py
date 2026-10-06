"""Lead-lag discovery and prediction (scope s.4 H1, s.10.1; tests T037-T039).

Positive lags always mean **X leads Y**. Correlation is discovery evidence only: a
relationship qualifies only if strictly-past X predicts future Y out of sample, beats the
zero and own-history baselines, and clears an economic hurdle.
"""

from cma.models.lead_lag.discovery import (
    HorizonEvaluation,
    LeadLagConfig,
    LeadLagResult,
    discover_lead_lag,
    lead_lag_grid,
    stratified_lead_lag,
)
from cma.models.lead_lag.estimators import (
    ContemporaneousCheck,
    CrossCorrelation,
    HYResult,
    PredictiveEvaluation,
    ScanSignificance,
    contemporaneous_check,
    cross_correlation,
    hayashi_yoshida,
    hy_lead_lag,
    scan_significance,
)
from cma.models.lead_lag.model import LeadLagPredictor, LeadLagPredictorConfig
from cma.models.lead_lag.series import (
    AlignedPair,
    EventSeries,
    ResampledSeries,
    ReturnKind,
    align_pair,
    grid_returns,
    make_grid,
    resample_locf,
)

__all__ = [
    "AlignedPair",
    "ContemporaneousCheck",
    "CrossCorrelation",
    "EventSeries",
    "HYResult",
    "HorizonEvaluation",
    "LeadLagConfig",
    "LeadLagPredictor",
    "LeadLagPredictorConfig",
    "LeadLagResult",
    "PredictiveEvaluation",
    "ResampledSeries",
    "ReturnKind",
    "ScanSignificance",
    "align_pair",
    "contemporaneous_check",
    "cross_correlation",
    "discover_lead_lag",
    "grid_returns",
    "hayashi_yoshida",
    "hy_lead_lag",
    "lead_lag_grid",
    "make_grid",
    "resample_locf",
    "scan_significance",
    "stratified_lead_lag",
]
