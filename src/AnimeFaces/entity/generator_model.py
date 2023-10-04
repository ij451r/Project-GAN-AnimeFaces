import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_size):
        self.latent_size = latent_size
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            
            # latent_size x 1 x 1
            nn.ConvTranspose2d(self.latent_size,512,kernel_size=4,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
        
            # 512 x 4 x 4
            nn.ConvTranspose2d(512,256,kernel_size=4,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            # 256 x 8 x 8
            nn.ConvTranspose2d(256,128,kernel_size=4,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # 128 x 16 x 16
            nn.ConvTranspose2d(128,64,kernel_size=4,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # 64 x 32 x 32
            nn.ConvTranspose2d(64,3,kernel_size=4,stride=2,padding=1,bias=False),
            nn.Tanh()
            
            #3 x 64 x 64
        )            
    def forward(self, x):
        return self.model(x)