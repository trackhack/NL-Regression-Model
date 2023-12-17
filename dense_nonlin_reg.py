from typing import Any
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional
import numpy as np

num_features = 3 

class dense_layer(pl.LightningModule):
    def __init__(self, prev_num_nodes, num_nodes):
        super(dense_layer, self).__init__()
        self.net= nn.Sequential(
            nn.BatchNorm1d(prev_num_nodes),
            nn.Linear(prev_num_nodes, num_nodes),
            nn.ReLU(inplace = True),
        )
        print(prev_num_nodes, num_nodes)
    def forward(self, x):
        out = self.net(x)
        return out

class dense_block(pl.LightningModule):
    def __init__(self, num_layers, num_inputs):
        super(dense_block, self).__init__()
        layers = []
        nodes = []
        scale_rate = 0.5

        #calc number of nodes per layer with a logarithmic decrease in nodes per layer
        for i in range (num_layers):
            nodes.append(np.ceil(num_inputs * np.exp(scale_rate*i)))
        nodes.append(num_inputs)
        nodes.reverse()

        #adding layers
        for i in range (num_layers):
            layers.append(dense_layer(prev_num_nodes = int(nodes[i]) ,num_nodes = int(nodes[i+1])))

        #construction block
        self.block = nn.Sequential(
            *layers,
            nn.Linear(num_inputs, 1),
            nn.BatchNorm1d(1),
            nn.ReLU(inplace = True)
            )
    
    def forward(self, x):
        out = self.block(x)
        out = torch.cat([out, x], dim = 1)  ### Verkettung von Output und Input und Speichern in Vektor
        return out 


class dense_regression_model(pl.LightningModule):
    def __init__(self, num_blocks = 5, num_layers = 5):
        super(dense_regression_model, self).__init__()
        #input layer
        self.input_layer = nn.Sequential(
            nn.Linear(num_features, num_features)
        )

        #creating of dense blocks
        blocks = []
        for i in range (num_blocks): 
            blocks.append(dense_block(num_layers = num_layers, num_inputs = num_features + i))
        self.blocks = nn.Sequential(*blocks)

        #output layer
        dim_out = num_features + num_blocks
        self.output_layer = nn.Sequential(
            nn.Linear(dim_out, 1),
            nn.BatchNorm1d(1),
            nn.ReLU(inplace = True)
        )

    def forward(self, x):
        x = self.input_layer(x)
        x = self.blocks(x)
        y = self.output_layer(x)
        return y

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),0.0001) 
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,1) 
        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx): 
        x, y = batch
        y_hat = self.forward(x)  
        y_hat = torch.flatten(y_hat)
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss, on_step = True, on_epoch = True, prog_bar = True, logger = True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        y_hat = torch.flatten(y_hat)
        loss = F.mse_loss(y_hat, y)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        y_hat = torch.flatten(y_hat)
        loss = F.mse_loss(y_hat, y)
        self.log("test_loss", loss, on_step = True, prog_bar = True)