from AnimeFaces import logger
from AnimeFaces.utils.device_utils import get_default_device, DeviceDataLoader, to_device
from AnimeFaces.utils.image_utils import show_batch
from AnimeFaces.entity.discriminator_model import Discriminator
from AnimeFaces.entity.generator_model import Generator

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from tqdm.notebook import tqdm


class TrainModel:
    def __init__(self, config: TrainModelConfig, device='cuda'):
        self.config = config
        self.device = get_default_device()
        self.generator = Generator(self.config.latent_size)
        self.generator.load_state_dict(torch.load(self.config.generator_model))
        self.discriminator = Discriminator()
        self.discriminator.load_state_dict(torch.load(self.config.discriminator_model))

    def load_models(self):
        self.generator = Generator(self.config.latent_size)
        self.generator.load_state_dict(torch.load(self.config.generator_model))
        self.generator = to_device(self.generator, self.device)
        self.discriminator = Discriminator()
        self.discriminator.load_state_dict(torch.load(self.config.discriminator_model))
        self.discriminator = to_device(self.discriminator, self.device)
        logger.info(f"Generator And Discriminator Model Loaded")

    def load_and_transform_data(self):
        DATA_DIR = self.config.image_source
        image_size = self.config.image_size
        batch_size = self.config.batch_size
        stats = (0.5,0.5,0.5),(0.5,0.5,0.5)
        train_ds = ImageFolder(DATA_DIR, transform=T.Compose([
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(*stats)
        ]))
        train_dl = DataLoader(train_ds, batch_size,shuffle=True, num_workers=2, pin_memory=True)
        self.train_dl = DeviceDataLoader(train_dl,self.device)
        show_batch(self.train_dl, self.config.root_dir)
        logger.info(f"Data Loaded to: {self.device} and saved to: {self.config.root_dir}")

    def train_discriminator(self,real_images,opt_d):
    
        opt_d.zero_grad()
        
        real_preds = self.discriminator(real_images)
        real_targets = torch.ones(real_images.size(0),1,device=self.device)
        real_loss = F.binary_cross_entropy(real_preds,real_targets)
        real_score = torch.mean(real_preds).item()
        
        latent = torch.randn(self.config.batch_size,self.config.latent_size,1,1,device=self.device)
        fake_images = self.generator(latent)
        
        fake_targets = torch.zeros(fake_images.size(0),1,device=self.device)
        fake_preds = self.discriminator(fake_images)
        fake_loss = F.binary_cross_entropy(fake_preds,fake_targets)
        fake_score = torch.mean(fake_preds).item()
        
        loss = real_loss + fake_loss
        loss.backward()
        opt_d.step()
        return loss.item(), real_score, fake_score

    def train_generator(self,opt_g):
        opt_g.zero_grad()
    
        latent = torch.randn(self.config.batch_size,self.config.latent_size,1,1,device=self.device)
        fake_images = self.generator(latent)
    
        preds = self.discriminator(fake_images)
        targets = torch.ones(self.config.batch_size,1,device=self.device)
        loss = F.binary_cross_entropy(preds,targets)
    
        loss.backward()
        opt_g.step()
    
        return loss.item()

    def fit(self,epochs,lr,start_idx=1):
        torch.cuda.empty_cache()
    
        losses_g=[]
        losses_d=[]
        real_scores = []
        fake_scores = []
    
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr,betas=(0.5,0.999))
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr,betas=(0.5,0.999))
    
        fixed_latent = torch.randn(64,latent_size,1,1,device=device)

        for epoch in range(epochs):
            for real_images,_ in tqdm(self.train_dl):
                loss_d , real_score , fake_score = self.train_discriminator(real_images,opt_d)
                loss_g = self.train_generator(opt_g)
                break
    
            losses_g.append(loss_g)
            losses_d.append(loss_d)
            real_scores.append(real_score)
            fake_scores.append(fake_score)
    
            logger.info("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}".format(epoch+1, epochs, loss_g, loss_d, real_score, fake_score))
            
            fake_images = self.generator_model(fixed_latent)
            save_samples("Training-Generated",epoch+start_idx,fake_images,self.config.root_dir, show)
            break
        return losses_g, losses_d, real_scores,fake_scores
    

    def save_trained_model(self, history):
        torch.save(self.discriminator_model.state_dict(), self.config.trained_discriminator_model)
        logger.info(f"Discrimator Model Trained And Saved at {self.config.trained_discriminator_model}")
        torch.save(self.generator_model.state_dict(), self.config.trained_generator_model)
        logger.info(f"Generator Model Trained And Saved at {self.config.trained_generator_model}")