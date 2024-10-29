from io import BytesIO
from PIL import ImageFile, Image
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
import random
from torchvision import datasets, models, transforms
from watermarkmodel.model.dataset import WatermarkDataset
import pandas as pd
ImageFile.LOAD_TRUNCATED_IMAGES = True

def Preprocessing(df_train, df_val):
    input_size = 256
    #normalization
    preprocess = {
        'train': transforms.Compose([
            transforms.Resize((input_size, input_size)),
            #transforms.RandomCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            RandomRotation([90, -90], 0.2),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((input_size, input_size)),
            #transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    #preprocessing
    train_ds = WatermarkDataset(df_train, preprocess['train'])
    val_ds = WatermarkDataset(df_val, preprocess['val'])
    datasets = {
        'train': train_ds,
        'val': val_ds,
    }
    return datasets

class RandomRotation:
    def __init__(self, angles, p):
        self.p = p
        self.angles = angles

    def __call__(self, x):
        if random.random() < self.p:
            angle = random.choice(self.angles)
            return transforms.functional.rotate(x, angle)
        else:
            return x


