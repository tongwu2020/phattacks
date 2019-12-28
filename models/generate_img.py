from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from core import Smooth
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import cv2
import torch.nn.functional as F
import torchfile
import numpy as np
from new_vgg_face import VGG_16
import shutil
import re
import argparse
from save_image import save_image



mean = [0.367035294117647,0.41083294117647057,0.5066129411764705]
data_transforms = {
    'test111': transforms.Compose([
        transforms.Resize(size = (224,224)),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, [1/255, 1/255, 1/255])
    ]),
    }
								
data_dir = '..'   # change this 
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['test111']}


dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1,
                                             shuffle=True)
              for x in ['test111']}


dataset_sizes = {x: len(image_datasets[x]) for x in ['test111']}
class_names = image_datasets['test111'].classes

print(class_names)
print(dataset_sizes)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument("model", type=str, help="test_model")
    args = parser.parse_args()
    

    model = VGG_16() 
    model.load_state_dict(torch.load('../donemodel/'+args.model))
    model.eval()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    correct = 0 
    total = 0 
    torch.manual_seed(12345)
    for data in dataloaders['test111']:
        images, labels = data
        images = images[:,[2,1,0],:,:]
            
        images = images.to(device)
        labels = labels.to(device)
        total += 1
        
        with torch.no_grad():
            model.eval()

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted.data == labels.data)
            print(predicted.data, labels.data)


        #if predicted.data == labels.data:
            #save_image('glass_uattack'+str(labels.data)+'_'+str(total),images.data)
            #save_image('glass_attack'+str(predicted.data)+'_'+str(total),Xadv.data)
            
           
    print( correct, total)
    # total = 0 

    # for data in dataloaders['test111']:
    #     total +=1
    #     images, labels = data
    #     images = images[:,[2,1,0],:,:]
    #     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
       
    #     images = images.to(device)
    #     labels = labels.to(device)
        
        
    #     model = VGG_16() 
    #     model.load_state_dict(torch.load('../donemodel/'+args.model))
    #     model.to(device)
        
    #     smoothed_classifier = Smooth(model, 10, 1)
        
    #     prediction = smoothed_classifier.predict(images, 1000, 0.001, 32)

    #     print(prediction,labels)
        
    #     if prediction != int(labels):
    #         save_image('11glass_uattack'+str(labels.data)+'__'+str(total),images.data)
    



    
    
    
    
    