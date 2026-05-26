from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    page_title: str = "AI Stock Forecaster"
    page_icon: str = "💹"
    layout: str = "wide"
    cache_ttl: str = "1d"


@dataclass(frozen=True)
class ModelConfig:
    input_size: int = 5
    hidden_size: int = 64
    num_layers: int = 2
    output_size: int = 5
    dropout: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 20
    sequence_length: int = 60
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    random_seed: int = 42
    artifacts_dir: str = "artifacts"
    market_cache_ttl_hours: int = 12
    max_download_retries: int = 4
    retry_backoff_seconds: float = 1.5


FEATURE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
CLOSE_COLUMN = "Close"
