# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 15:23:12 2020

@author: zjermain15
"""
import matplotlib.pyplot as plt
import torch 
device = torch.device('cuda')

from experiments.dataloaders import ConcentricSphere
from torch.utils.data import DataLoader 
from viz.plots import single_feature_plt 

data_dim = 2 
data_concentric = ConcentricSphere(data_dim, inner_range=(0.,.5), outer_range = (1., 1.5),num_points_inner=1000, num_points_outer=2000)

dataloader = DataLoader(data_concentric, batch_size=64, shuffle=True)

# Visualize a batch of data (use a large batch size for visualization)
dataloader_viz = DataLoader(data_concentric, batch_size=256, shuffle=True)
for inputs, targets in dataloader_viz:
    break

single_feature_plt(inputs, targets)

from anode.models import ODENet 
from anode.training import Trainer 

hidden_dim = 32

model = ODENet(device, data_dim, hidden_dim,time_dependent = True, non_linearity = 'relu')
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

from viz.plots import get_feature_history

trainer = Trainer(model,optimizer,device)
num_epochs = 10

visualize_features = True 

if visualize_features: 
    feature_history = get_feature_history(trainer, dataloader, inputs, targets, num_epochs)
else: 
    trainer.train(dataloader,num_epochs)

from viz.plots import multi_feature_plt

multi_feature_plt(feature_history[::2], targets)

from viz.plots import trajectory_plt

# To make the plot clearer, we will use a smaller batch of data
for small_inputs, small_targets in dataloader:
    break

trajectory_plt(model, small_inputs, small_targets, timesteps=10)

from viz.plots import input_space_plt

input_space_plt(model)


plt.plot(trainer.histories['loss_history'])
plt.xlim(0, len(trainer.histories['loss_history']) - 1)
plt.ylim(0)
plt.xlabel('Iterations')
plt.ylabel('Loss')


plt.plot(trainer.histories['nfe_history'])
plt.xlim(0, len(trainer.histories['nfe_history']) - 1)
plt.ylim(0)
plt.xlabel('Iterations')
plt.ylabel('NFEs')
