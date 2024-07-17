import sys
import os
import torch

from torch import nn
from torch.utils.data import DataLoader 

from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
from dataset.preprocessing_model_2 import CustomDataset_Conditional
from models.unet_sa import UNet_conditional
from models.ddpm import Diffusion

# Set seed for PyTorch
torch.manual_seed(42)


# Read the configuration
config_name = "default"
pipe = ConfigPipeline(
    [
        YamlConfig(
            "./conditional_ddpm_config.yaml", config_name='default', config_folder='cfg/'
        ),
        ArgparseConfig(infer_types=True, config_name=None, config_file=None),
        YamlConfig(config_folder='cfg/')
    ]
)
config = pipe.read_conf()
#pipe.log()
#print(config.unet.input_channels)

# Load the Dataset
#dataset = CustomDataset_Conditional(folder_path=config.data.folder)
#data_loader = DataLoader(dataset=dataset, batch_size=config.data.batch_size, shuffle=config.data.shuffle)

# Load model
model = UNet_conditional(config)
model.load_state_dict(torch.load('saved_models/new_conditional_ckpt_model2.pt'))#, map_location=torch.device('cpu'))
model = model.to(device=config.device)


# Load diffusion 
diffusion = Diffusion(noise_steps=config.diff.noise_steps, beta_start=config.diff.beta_start, beta_end=config.diff.beta_end, img_size=config.data.image_size, device=config.device)
#labels = torch.ones([8,], dtype=torch.long).to(config.device)


# Sampling Images
# labels = torch.zeros([8,], dtype=torch.long).to(config.device)
# sampled_images = diffusion.sample_conditional(model, n=8, labels=labels)
# diffusion.save_images(sampled_images, os.path.join("plots", f"no_sub_420.jpg"))

# Calculating Fid scores
diffusion.cal_fid_all(model=model, data_dir=config.data.folder, device=config.device)
