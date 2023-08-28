import datetime

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
import torch.nn.functional as F
import timm
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger

from dataset_pr import PoseRegressionDataset


class PoseRegressionModel(LightningModule):
    def __init__(self, image_size):
        super(PoseRegressionModel, self).__init__()
        # Create EfficientNet-B0 with pretrained weights
        efficientnet_pretrained = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)

        # Get the weights from the first convolutional layer
        pretrained_weights = efficientnet_pretrained.conv_stem.weight

        # Average the weights across the three input channels
        new_weights = pretrained_weights.mean(dim=1, keepdim=True)

        # Create a new convolutional layer with one input channel
        new_conv = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)

        # Set the weights of the new convolutional layer
        new_conv.weight = nn.Parameter(new_weights)

        # Replace the first convolutional layer in the model
        efficientnet_pretrained.conv_stem = new_conv

        self.efficientnet = efficientnet_pretrained

        # Get the output feature size of EfficientNet-B0
        num_features = self.efficientnet.num_features

        # translation normalization coeff
        self.translation_normalization_coeff = np.array(image_size).mean()/2

        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 7)
        )

    def geodesic_loss(self, q1, q2):
        # Calculate the dot product between the two quaternions
        dot_product = (q1 * q2).sum(dim=1)
        # Clamp the values to avoid numerical instability
        dot_product = torch.clamp(dot_product, -1.0, 1.0)
        # Calculate the angle between the two quaternions
        angle = torch.acos(dot_product)
        return angle.mean()

    def forward(self, x):
        x = self.efficientnet(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        # Normalize the quaternion part
        # x[:, :4] = F.normalize(x[:, :4], p=2, dim=1)
        return x

    def training_step(self, batch, batch_idx):
        images, poses = batch
        predicted_poses = self(images)
        normalized_rot = F.normalize(predicted_poses[:, :4], p=2, dim=1)
        loss_rot = self.geodesic_loss(normalized_rot, poses[:, :4])
        loss_trans = F.mse_loss(predicted_poses[:, 4:], poses[:, 4:] / self.translation_normalization_coeff)
        loss = loss_rot + loss_trans
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return loss

    # Create a ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',  # Change this to the metric you want to monitor
        dirpath='../checkpoints/pose_regression/',  # Directory where the checkpoints will be saved
        filename="best_model-{epoch:02d}-{val_loss:.2f}.pt",  # File naming scheme
        save_top_k=1,  # Only save the top k models (in this case, just the best one)
        mode='min'  # Save the model with the minimum validation loss
    )

    def validation_step(self, batch, batch_idx):
        images, poses = batch
        predicted_poses = self(images)
        normalized_rot = F.normalize(predicted_poses[:, :4], p=2, dim=1)
        loss_rot = self.geodesic_loss(normalized_rot, poses[:, :4])
        loss_trans = F.mse_loss(predicted_poses[:, 4:], poses[:, 4:] / self.translation_normalization_coeff)
        loss = loss_rot + loss_trans
        self.log('val_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)


# Define your data paths and parameters
csv_file = '/home/farid/Desktop/phantom_simulated_data/tracking.csv'
us_image_dir = '/home/farid/Desktop/phantom_simulated_data/2d_images/'
ct_image_path = '/home/farid/Desktop/phantom_simulated_data/ct_seg.mhd'

# Create the dataset
dataset = PoseRegressionDataset(csv_file, us_image_dir, ct_image_path)

# Split the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Initialize the model
model = PoseRegressionModel(dataset.ct_image.GetSize())

# Create a TensorBoard logger
logger = TensorBoardLogger('../checkpoints/pose_regression/',
                           name=f'pose_regression_{datetime.datetime.now().strftime("%Y.%m.%d_%H.%M")}')
# Create a WandB logger
wandb_logger = WandbLogger(project='pose regression')

# Initialize the Trainer with the logger
trainer = pl.Trainer(max_epochs=20, logger=[logger, wandb_logger], callbacks=[model.checkpoint_callback], gpus=1)

# Start training
trainer.fit(model, train_loader, val_loader)