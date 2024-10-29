import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
from PIL import Image
import pandas as pd
import random
from tqdm import tqdm
import timm
import pickle
import sys
sys.path.append('../')
from watermarkmodel.model.convnext import convnext_tiny
from watermarkmodel.model.dataset import WatermarkDataset
from watermarkmodel.model.preprocess import RandomRotation
from watermarkmodel.model.train import train_model
from watermarkmodel.model.preprocess import Preprocessing

def train_ml_model():
    df_train = pd.read_csv('../dataset/train_data_v1.csv')
    df_val = pd.read_csv('../dataset/val_data_v1.csv')
    BATCH_SIZE = 8
    lrate = 0.1e-3 
    epoch = 10

    datasets = Preprocessing(df_train, df_val)

    model_ft = convnext_tiny(pretrained=True, in_22k=True, num_classes=21841)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #config
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.AdamW(params=model_ft.parameters(), lr=lrate)
    dataloaders_dict = {
        x: torch.utils.data.DataLoader(datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=0) #to prevent runtimeerror on non gpu device
        for x in ['train', 'val']
    }

    #train
    model_ft.head = nn.Sequential( 
        nn.Linear(in_features=768, out_features=512),
        nn.GELU(),
        nn.Linear(in_features=512, out_features=256),
        nn.GELU(),
        nn.Linear(in_features=256, out_features=2),
    )

    model_ft, train_acc_history, val_acc_history = train_model(
        model_ft, dataloaders_dict, criterion, optimizer, num_epochs=epoch
    )

    #save model
    filename = 'watermark_model.pkl'
    return (pickle.dump(model_ft, open(filename, 'wb')))