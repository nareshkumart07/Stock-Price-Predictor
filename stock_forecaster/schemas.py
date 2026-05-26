from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


@dataclass(frozen=True)
class CompanyProfile:
    name: str
    sector: str
    logo_url: str


@dataclass(frozen=True)
class StockBundle:
    data: pd.DataFrame
    last_window: pd.DataFrame
    profile: CompanyProfile


@dataclass(frozen=True)
class ForecastResult:
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: float
    delta_close: float
    trend: str


@dataclass(frozen=True)
class RegressionMetrics:
    mae: float
    rmse: float
    mape: float


@dataclass(frozen=True)
class EvaluationSummary:
    model_metrics: RegressionMetrics
    baseline_metrics: RegressionMetrics
    direction_accuracy: float
    train_size: int
    val_size: int
    test_size: int


@dataclass
class TrainingArtifact:
    model: Any
    scaler: MinMaxScaler
    evaluation: EvaluationSummary
    run_config: Dict[str, Any]
    source: str
    artifact_path: str
