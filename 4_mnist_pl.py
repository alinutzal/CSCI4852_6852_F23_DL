import os
import lightning.pytorch as pl

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torchmetrics
import wandb
from lightning.pytorch.loggers import WandbLogger

#import torch.profiler as profiler 
from torch.profiler import tensorboard_trace_handler

PATH_DATASETS = os.environ.get("PATH_DATASETS","/users/PLS0129/ysu0053/CSCI4852_6852_F23_DL/data")
BATCH_SIZE = 1024 if torch.cuda.is_available() else 64

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
    def __init__(self, data_dir=PATH_DATASETS, hidden_size=64, learning_rate=2e-4):
        super().__init__()

        # Set our init args as class attributes
        self.data_dir = data_dir
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        # Hardcode some dataset specific attributes
        self.num_classes = 10
        self.dims = (1, 28, 28)
        channels, width, height = self.dims
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        # Define PyTorch model
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * width * height, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, self.num_classes),
        )

        self.train_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=10)
        self.val_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=10)
        self.test_accuracy = torchmetrics.classification.Accuracy(task="multiclass", num_classes=10)

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.train_accuracy.update(preds, y)
        self.log("loss", loss, prog_bar=True, sync_dist=True)
        self.log("train_acc", self.train_accuracy, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy.update(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_accuracy.update(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("test_loss", loss, prog_bar=True,sync_dist=True)
        self.log("test_acc", self.test_accuracy, prog_bar=True,sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate*2)
        return optimizer

    ####################
    # DATA RELATED HOOKS
    ####################

    def prepare_data(self):
        # download
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = datasets.MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(mnist_full, [50000, 10000])

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = datasets.MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=BATCH_SIZE, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=BATCH_SIZE, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=BATCH_SIZE, num_workers=4)

class TorchTensorboardProfilerCallback(pl.Callback):
  """Quick-and-dirty Callback for invoking TensorboardProfiler during training.
  
  For greater robustness, extend the pl.profiler.profilers.BaseProfiler. See
  https://pytorch-lightning.readthedocs.io/en/stable/advanced/profiler.html"""

  def __init__(self, profiler):
    super().__init__()
    self.profiler = profiler 

  def on_train_batch_end(self, trainer, pl_module, outputs, *args, **kwargs):
    self.profiler.step()
    pl_module.log_dict(outputs)  # also logging the loss, while we're here

# initial values are defaults, for all except batch_size, which has no default
config = {"batch_size": 32,  # try log-spaced values from 1 to 50,000
          "num_workers": 0,  # try 0, 1, and 2
          "pin_memory": False,  # try False and True
          "precision": 32,  # try 16 and 32
          "optimizer": "Adadelta",  # try optim.Adadelta and optim.SGD
          }

with wandb.init(project="trace", config=config) as run:

    # Set up MNIST data
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    dataset = datasets.MNIST("../data", train=True, download=True,
                            transform=transform)

    ## Using a raw DataLoader, rather than LightningDataModule, for greater transparency
    trainloader = torch.utils.data.DataLoader(
      dataset,
      # Key performance-relevant configuration parameters:
      ## batch_size: how many datapoints are passed through the network at once?
      batch_size=wandb.config.batch_size,
      # larger batch sizes are more compute efficient, up to memory constraints

      ##  num_workers: how many side processes to launch for dataloading (should be >0)
      num_workers=wandb.config.num_workers,
      # needs to be tuned given model/batch size/compute

      ## pin_memory: should a fixed "pinned" memory block be allocated on the CPU?
      pin_memory=wandb.config.pin_memory,
      # should nearly always be True for GPU models, see https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/
      )
    
    # Set up model
    model = MNISTLightning(optimizer=wandb.config["optimizer"])

    # Set up profiler
    wait, warmup, active, repeat = 1, 1, 2, 1
    total_steps = (wait + warmup + active) * (1 + repeat)
    schedule =  torch.profiler.schedule(
      wait=wait, warmup=warmup, active=active, repeat=repeat)
    profiler = torch.profiler.profile(
      schedule=schedule, on_trace_ready=tensorboard_trace_handler("wandb/latest-run/tbprofile"), with_stack=False)

    with profiler:
        profiler_callback = TorchTensorboardProfilerCallback(profiler)

        trainer = pl.Trainer(gpus=1, max_epochs=1, max_steps=total_steps,
                            logger=pl.loggers.WandbLogger(log_model=True, save_code=True),
                            callbacks=[profiler_callback], precision=wandb.config.precision)

        trainer.fit(model, trainloader)

    profile_art = wandb.Artifact(f"trace-{wandb.run.id}", type="profile")
    profile_art.add_file(glob.glob("wandb/latest-run/tbprofile/*.pt.trace.json")[0], "trace.pt.trace.json")
    run.log_artifact(profile_art)