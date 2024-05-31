import sys
import torch

from torch import nn
from torch.utils.data import DataLoader 

from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
from dataset.preprocessing_model_2 import CustomDataset_Conditional
from models.unet_sa import UNet_conditional
from train.train_conditional_ddpm import Trainer

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
dataset = CustomDataset_Conditional(folder_path=config.data.folder)
data_loader = DataLoader(dataset=dataset, batch_size=config.data.batch_size, shuffle=config.data.shuffle)

# Load model
model = UNet_conditional(config)
model = model.to(device=config.device)

# Create Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.opt.lr,
    eps=config.opt.eps,
)

if config.opt.scheduler == "OneCycleLR":
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr = config.opt.lr,
        steps_per_epoch = len(data_loader),
        epochs = config.opt.epochs,
    ) 
elif config.opt.scheduler == "ReduceLROnPlateau":
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=config.opt.gamma,
        patience=config.opt.scheduler_patience,
        mode="min",
    )
elif config.opt.scheduler == "CosineAnnealingLR":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.opt.scheduler_T_max
    )
elif config.opt.scheduler == "StepLR":
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=config.opt.step_size, gamma=config.opt.gamma
    )
else:
    raise ValueError(f"Got scheduler={config.opt.scheduler}")


# Create the loss
mse = nn.MSELoss()

trainer = Trainer(model=model, config=config)

if config.verbose:
    print("\n### MODEL ###\n", print(model))
    print("\n### OPTIMIZER ###\n", optimizer)
    print("\n### SCHEDULAR ###\n", scheduler)
    print("\n### LOSSES ###\n")
    print("\n### Beginning Training ...\n")


# Train
trainer.train(data_loader=data_loader, mse=mse, optimizer=optimizer, scheduler=scheduler)