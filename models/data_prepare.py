	
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import os
import copy
import cv2
import torch.nn.functional as F
import torchfile
import numpy as np
import argparse
from time import time
import datetime

def datapre():
    data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(size = (224,224)), #(size=(224, 224)
        #transforms.RandomHorizontalFlip(),
        #transforms.RandomCrop(size=(224, 224)),
        transforms.ToTensor(),

        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(size = (224,224)), #(size=(224, 224)
        #transforms.RandomHorizontalFlip(),
        #transforms.RandomCrop(size=(224, 224)),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(size = (224,224)),
        transforms.ToTensor(),
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
	}


    
    data_dir = '/home/research/tongwu/glass'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val','test']}
    
    
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1,
                                                  shuffle=True)
                   for x in ['train', 'val','test']}
    
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val','test']}
    class_names = image_datasets['train'].classes
    print(class_names)
    print(dataset_sizes)


    return dataloaders, dataset_sizes
    
