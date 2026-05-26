from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch

from stock_forecaster.config import ModelConfig
from stock_forecaster.schemas import EvaluationSummary, RegressionMetrics, TrainingArtifact
from stock_forecaster.training_service import build_model, train_model


def _safe_ticker_name(ticker: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in ticker.upper())


def get_artifact_path(ticker: str, config: ModelConfig) -> Path:
    base = Path(__file__).resolve().parents[1] / config.artifacts_dir
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{_safe_ticker_name(ticker)}_lstm.pt"


def save_artifact(
    ticker: str,
    artifact_path: Path,
    model,
    scaler,
    evaluation: EvaluationSummary,
    run_config,
) -> None:
    payload = {
        "ticker": ticker,
        "model_state_dict": model.state_dict(),
        "scaler": scaler,
        "evaluation": asdict(evaluation),
        "run_config": run_config,
    }
    torch.save(payload, artifact_path)


def _deserialize_evaluation(raw_eval: dict) -> EvaluationSummary:
    model_metrics = RegressionMetrics(**raw_eval["model_metrics"])
    baseline_metrics = RegressionMetrics(**raw_eval["baseline_metrics"])
    return EvaluationSummary(
        model_metrics=model_metrics,
        baseline_metrics=baseline_metrics,
        direction_accuracy=float(raw_eval["direction_accuracy"]),
        train_size=int(raw_eval["train_size"]),
        val_size=int(raw_eval["val_size"]),
        test_size=int(raw_eval["test_size"]),
    )


def load_artifact(ticker: str, config: ModelConfig) -> Optional[TrainingArtifact]:
    artifact_path = get_artifact_path(ticker, config)
    if not artifact_path.exists():
        return None

    payload = torch.load(artifact_path, map_location="cpu")
    model = build_model(config)
    model.load_state_dict(payload["model_state_dict"])

    evaluation = _deserialize_evaluation(payload["evaluation"])

    return TrainingArtifact(
        model=model,
        scaler=payload["scaler"],
        evaluation=evaluation,
        run_config=payload.get("run_config", {}),
        source="loaded",
        artifact_path=str(artifact_path),
    )


def get_or_train_artifact(
    ticker: str, data, config: ModelConfig, force_retrain: bool = False
) -> TrainingArtifact:
    if not force_retrain:
        cached = load_artifact(ticker, config)
        if cached is not None:
            return cached

    model, scaler, evaluation, run_config = train_model(data, config)
    artifact_path = get_artifact_path(ticker, config)
    save_artifact(
        ticker=ticker,
        artifact_path=artifact_path,
        model=model,
        scaler=scaler,
        evaluation=evaluation,
        run_config=run_config,
    )

    return TrainingArtifact(
        model=model,
        scaler=scaler,
        evaluation=evaluation,
        run_config=run_config,
        source="trained",
        artifact_path=str(artifact_path),
    )
