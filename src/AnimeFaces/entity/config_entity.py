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

@dataclass(frozen=True)
class TrainModelConfig:
    root_dir: Path
    generator_model: Path
    discriminator_model: Path
    trained_generator_model: Path
    trained_discriminator_model: Path
    image_source: Path
    latent_size: int
    image_size: int
    batch_size: int    

@dataclass(frozen=True)
class GenerateConfig:
    root_dir: Path
    trained_generator_model: Path
    trained_discriminator_model: Path    
    latent_size: int