import torch.nn as nn
import torch
from AnimeFaces import logger
from AnimeFaces.utils.device_utils import get_default_device
from AnimeFaces.utils.image_utils import save_samples
from AnimeFaces.entity.config_entity import PrepareModelConfig
from AnimeFaces.entity.generator_model import Generator
from AnimeFaces.entity.discriminator_model import Discriminator

class PrepareModel:
    def __init__(self, config: PrepareModelConfig):
        self.config = config
        self.device = get_default_device()
        
    def PrepareDiscriminatorModel(self):
        self.discriminator_model = Discriminator()
        torch.save(self.discriminator_model.state_dict(), self.config.discriminator_model)
        logger.info(f"Discrimator Model Created And Saved at {self.config.discriminator_model}")

    def PrepareGeneratorModel(self):
        self.generator_model = Generator(self.config.latent_size)
        torch.save(self.generator_model.state_dict(), self.config.generator_model)
        logger.info(f"Generator Model Created And Saved at {self.config.generator_model}")
        
    def TestGeneratorModel(self, show=True):
        fixed_latent = torch.randn(64,self.config.latent_size,1,1)
        fake_images = self.generator_model(fixed_latent)
        save_samples("TestGeneratorModel",0,fake_images,self.config.root_dir, show)