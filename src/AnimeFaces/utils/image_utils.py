import torch
from torchvision.utils import make_grid
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import os
from AnimeFaces import logger

def denorm(img_tensors):
  Stat = ((0.5,0.5,0.5),(0.5,0.5,0.5))
  return img_tensors*Stat[1][0]+Stat[0][0]

def save_samples(name,index,images,save_dir,show=True):
  image_name = '{name}-{index:04d}.png'.format(name=name, index=index)
  denorm_images = denorm(images)
  save_image(denorm_images,os.path.join(save_dir,image_name),nrow=8)
  logger.info(f"Sample Generated Images Saved: {os.path.join(save_dir,image_name)}")
  if show:
    fig,ax=plt.subplots(figsize=(8,8))
    ax.set_xticks([]);ax.set_yticks([])
    ax.imshow(make_grid(denorm_images.cpu().detach(),nrow=8).permute(1,2,0))
    plt.show()

def show_batch(dl,save_dir):
  for images,_ in dl:
    save_samples("loaded-images",1,images,save_dir)
    break  