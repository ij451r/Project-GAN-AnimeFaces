from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    kaggle_source: str
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class PrepareModelConfig:
    root_dir: Path
    generator_model: Path
    discriminator_model: Path
    latent_size: int