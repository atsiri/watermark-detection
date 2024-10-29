from io import BytesIO
from PIL import ImageFile, Image
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
import random
from torchvision import datasets, models, transforms
ImageFile.LOAD_TRUNCATED_IMAGES = True

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


