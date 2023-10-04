import torch
from AnimeFaces import logger
from AnimeFaces.utils.device_utils import get_default_device, to_device
from AnimeFaces.utils.image_utils import save_samples
from AnimeFaces.entity.config_entity import PrepareModelConfig
from AnimeFaces.entity.generator_model import Generator

class Generate:
	def __init__(self, config: PrepareModelConfig):
		self.config = config
		self.device = get_default_device()

	def generate_image(self, show=True):
		latent_noise = torch.randn(9,self.config.latent_size,1,1,device=self.device)
		generator_model = Generator(self.config.latent_size)
		generator_model.load_state_dict(torch.load(self.config.trained_generator_model))
		generator_model = to_device(generator_model,self.device)
		fake_images = generator_model(latent_noise)
		logger.info(f"Fake Images Generated")
		save_samples("TrainedGeneratorModel",0,fake_images,self.config.root_dir, show)

