# this file is based on https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
# Author: Sasank Chilamkurthy

'''
This file basically runs Defending against Circle Occlusion Attacks (DOA) see paper 
Type 'python sticker_retrain.py {}.pt -alpha 4 -iters 50 -out 99 -search 1 -epochs 5' to run 
{}.pt is the name of model you want to train with DOA 
alpha is learning rate of PGD e.g. 4
iters is the iterations of PGD e.g.50
out is name of your final model e.g.99
search is method of searching, '0' is exhaustive_search, '1' is gradient_based_search"
epochs is the epoch you want to fine tune your network e.g. 5

Note that COA is a abstract attacking model simulate the "physical" attacks, which is a circle
Thus there is no restriction for the mask to be rectangle
'''


import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import time
from torchvision import datasets, models, transforms
import cv2
import torchfile
import matplotlib.pyplot as plt
from origin_train import data_process
import argparse
import copy
from origin_test import test
from new_vgg_face import VGG_16
from Oshape_attack import COA
#from save_image import *


def sticker_train_model(model, criterion, optimizer, scheduler, alpha, iters, search, num_epochs=10):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    circle = cv2.imread('./dataprepare/circle.jpg') 
    circle = transforms.ToTensor()(circle)
    circle = circle.to(device)
    circle[circle>=0.5] = 1
    circle[circle<0.5] = 0
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs[:,[2,1,0],:,:] #rgb to bgr
                COA_module = COA(model,alpha,iters)
                with torch.set_grad_enabled(search==1):
                    if search == 0:
                        COA_inputs = COA_module.exhaustive_search(inputs, labels,args.width,args.height,args.stride,args.stride,circle)
                        #save_image("COA_retain_noised_image",COA_inputs.data)
                        # if the number is wrong, run gradient_based_search, since this method can save time
                    else:
                        COA_inputs = COA_module.gradient_based_search(inputs, labels,args.width,args.height,args.stride,args.stride,circle)
                        #save_image("COA_retain_noised_image2",COA_inputs.data)

                optimizer.zero_grad()
                if phase == 'train':
                    model.train()
                    
                with torch.set_grad_enabled(phase == 'train'):
                    
                    outputs = model(COA_inputs)
                    _, preds = torch.max(outputs, 1)
                    labels = labels.to(device)
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # statistics
                running_loss += loss.item() * inputs.size(0)

                running_corrects += torch.sum(preds == labels.data)
                print("correct", running_corrects)
                
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model_ft.state_dict(), '../donemodel/new_sticker_model0'+str(args.out)+'.pt')
        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


    
if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser(description='Predict on many examples')
    parser.add_argument("model", type=str, help="original(clean) model you want to do DOA ")
    parser.add_argument("-alpha", type=int, help="alpha leanrning rate")
    parser.add_argument("-iters", type=int, help="iterations of PGD ")
    parser.add_argument("-out", type=int, help="name of final model")
    parser.add_argument("-search", type=int, help="method of searching, \
        '0' is exhaustive_search, '1' is gradient_based_search")
    parser.add_argument("-epochs", type=int, help="epochs")
    parser.add_argument("--stride", type=int, default=10, help="the skip pixels when searching")
    parser.add_argument("--width", type=int, default= 80, help="width of the cirlce occlusion")
    parser.add_argument("--height", type=int, default=80, help="height of the cirlce occlusion")
    args = parser.parse_args()
    
    print(args)
    
    
    torch.manual_seed(123456)
    torch.cuda.empty_cache()
    print('output model will locate on ../donemodel/new_sticker_model0'+str(args.out)+'.pt')
    dataloaders,dataset_sizes =data_process(batch_size =32)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = VGG_16() 
    model_ft.load_state_dict(torch.load('../donemodel/'+args.model))
    #model_ft.load_weights()
    model_ft.to(device)
    
    # model_ft = nn.DataParallel(model,device_ids=[0,1])
    
    criterion = nn.CrossEntropyLoss()
    
    #optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.0001)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)

    
    model_ft = sticker_train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,args.alpha, args.iters,args.search, num_epochs=args.epochs)
    test(model_ft,dataloaders,dataset_sizes)

    torch.save(model_ft.state_dict(), '../donemodel/new_sticker_model0'+str(args.out)+'.pt')



