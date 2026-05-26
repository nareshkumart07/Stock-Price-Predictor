import numpy as np
from sklearn.preprocessing import MinMaxScaler

from stock_forecaster.config import ModelConfig
from stock_forecaster.model_store import load_artifact, save_artifact
from stock_forecaster.schemas import EvaluationSummary, RegressionMetrics
from stock_forecaster.training_service import build_model


def test_model_store_save_and_load_roundtrip(tmp_path):
    config = ModelConfig(artifacts_dir=str(tmp_path))
    artifact_path = tmp_path / "TEST_lstm.pt"

    model = build_model(config)
    scaler = MinMaxScaler().fit(np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6]], dtype=float))

    evaluation = EvaluationSummary(
        model_metrics=RegressionMetrics(mae=1.0, rmse=2.0, mape=3.0),
        baseline_metrics=RegressionMetrics(mae=1.5, rmse=2.5, mape=3.5),
        direction_accuracy=55.0,
        train_size=10,
        val_size=5,
        test_size=4,
    )

    run_config = {"epochs": 5.0, "seed": 42.0}

    save_artifact(
        ticker="TEST",
        artifact_path=artifact_path,
        model=model,
        scaler=scaler,
        evaluation=evaluation,
        run_config=run_config,
    )

    loaded = load_artifact("TEST", config)

    assert loaded is not None
    assert loaded.source == "loaded"
    assert loaded.evaluation.model_metrics.mae == 1.0
    assert loaded.evaluation.baseline_metrics.rmse == 2.5
    assert loaded.run_config["seed"] == 42.0
