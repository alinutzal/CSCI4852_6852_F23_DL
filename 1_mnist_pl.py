import os
import lightning.pytorch as pl

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import wandb
from lightning.pytorch.loggers import WandbLogger

PATH_DATASETS = os.environ.get("PATH_DATASETS","/users/PLS0129/ysu0053/CSCI4852_6852_F23_DL/data")
BATCH_SIZE = 256 if torch.cuda.is_available() else 64

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print('Number of GPUs:',torch.cuda.device_count())
print()

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
    
class MNISTModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

# Init our model
mnist_model = MNISTModel()

# Init DataLoader from MNIST Dataset
train_ds = datasets.MNIST(PATH_DATASETS, train=True, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,num_workers=2)

# Initialize wandb
wandb.init(project='mnist_mlp')
settings=wandb.Settings(silent="True")

# Create the WandbLogger
wandb_logger = WandbLogger()

# Initialize a trainer
trainer = pl.Trainer(accelerator="gpu", devices=2, strategy="ddp", max_epochs=20, logger=wandb_logger) #    accelerator="auto",devices=1,

# Train the model ⚡
trainer.fit(mnist_model, train_loader)

# Close wandb run
wandb.finish()