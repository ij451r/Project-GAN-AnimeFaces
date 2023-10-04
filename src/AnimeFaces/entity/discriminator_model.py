import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            # Layer 1: 3 x 64 x 64 -> 64 x 32 x 32
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 2: 64 x 32 x 32 -> 128 x 16 x 16
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 3: 128 x 16 x 16 -> 256 x 8 x 8
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 4: 256 x 8 x 8 -> 512 x 4 x 4
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 5: 512 x 4 x 4 -> 1 x 1 x 1
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
            
            # Flatten for the final layer
            nn.Flatten(),
            
            # Sigmoid activation for binary classification
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)