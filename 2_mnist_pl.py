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
    
# Define the Lightning module
class MNISTLightning(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('val_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(mnist_train, batch_size=32, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(mnist_val, batch_size=32)
    
    # Add a test dataloader
    def test_dataloader(self):
        return torch.utils.data.DataLoader(mnist_test, batch_size=32, shuffle=False)
    
# Init our model
mnist_model = MNISTLightning()

# Init DataLoader from MNIST Dataset
mnist_full = datasets.MNIST(PATH_DATASETS, train=True, download=True, transform=transforms.ToTensor())
mnist_train, mnist_val = random_split(mnist_full, [50000, 10000])
#train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,num_workers=2)
#val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,num_workers=2)

mnist_test = datasets.MNIST(PATH_DATASETS, train=False, download=True, transform=transforms.ToTensor())
#test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE,num_workers=2)

# Initialize wandb
wandb.init(project='mnist_mlp')
settings=wandb.Settings(silent="True")

# Create the WandbLogger
wandb_logger = WandbLogger()

# Initialize a trainer
trainer = pl.Trainer(accelerator="gpu", devices=2, strategy="ddp", max_epochs=20, logger=wandb_logger) #    accelerator="auto",devices=1,

# Train the model ⚡
trainer.fit(mnist_model)

#Test
trainer.test()
# Close wandb run
wandb.finish()