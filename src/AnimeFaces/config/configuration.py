from AnimeFaces.constants import * 
from AnimeFaces.utils.common import read_yaml, create_directories
from AnimeFaces.entity.config_entity import (
	DataIngestionConfig,
    PrepareModelConfig,
    TrainModelConfig,
    GenerateConfig,
)

class ConfigurationManager:
    def __init__(self , config_filepath = CONFIG_FILE_PATH , params_filepath = PARAMS_FILE_PATH , schema_filepath = SCHEMA_FILE_PATH,):
        
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)
        
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])
        data_ingestion_config = DataIngestionConfig(
            root_dir = config.root_dir,
            kaggle_source = config.kaggle_source,
            source_URL = config.source_URL,
            local_data_file = config.local_data_file,
            unzip_dir = config.unzip_dir
        )
        return data_ingestion_config

    def get_prepare_model_config(self) -> PrepareModelConfig:
        config = self.config.prepare_model
        create_directories([config.root_dir])
        prepare_model_config = PrepareModelConfig(
            root_dir = config.root_dir,
            generator_model = config.generator_model,
            discriminator_model = config.discriminator_model,
            latent_size = config.latent_size,
        )
        return prepare_model_config        

    def get_train_model_config(self) -> TrainModelConfig:
        config = self.config.train_model
        create_directories([config.root_dir])
        train_model_config = TrainModelConfig(
            root_dir = config.root_dir,
            generator_model = config.generator_model,
            discriminator_model = config.discriminator_model,
            trained_generator_model = config.trained_generator_model,
            trained_discriminator_model = config.trained_discriminator_model,
            image_source = config.image_source,
            latent_size = config.latent_size,
            image_size = config.image_size,
            batch_size = config.batch_size,
        )
        return train_model_config        

    def get_generate_config(self) -> GenerateConfig:
        config = self.config.generate
        create_directories([config.root_dir])
        generate_config = GenerateConfig(
            root_dir = config.root_dir,
            trained_generator_model = config.trained_generator_model,
            trained_discriminator_model = config.trained_discriminator_model,
            latent_size = config.latent_size,
        )
        return generate_config
