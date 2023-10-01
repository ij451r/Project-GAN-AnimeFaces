import os
import urllib.request as request
import zipfile
from AnimeFaces import logger
from AnimeFaces.utils.common import get_size
import kaggle
from AnimeFaces.entity.config_entity import DataIngestionConfig
from pathlib import Path

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            DATASET_ORIGINAL_SIZE = 404127
            logger.info(f"Files from URL: {self.config.source_URL} will be downloaded and zipped at {self.config.local_data_file}")
            file = kaggle.api.dataset_download_files(self.config.kaggle_source, path=self.config.unzip_dir)
            logger.info(f"File Has Been Downloaded with size {get_size(Path(self.config.local_data_file))}")
        else:
            logger.info(f"File already existts of size: {get_size(Path(self.config.local_data_file))}")
    def extract_zip_file(self):
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)     