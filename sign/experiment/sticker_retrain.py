# this file is based on https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
# Author: Sasank Chilamkurthy

'''
This file basically runs Defending against rectangular Occlusion Attacks (DOA) see paper 
Type 'python sticker_retrain.py {}.pt -alpha 0.01 -iters 30 -out 99 -search 1 -epochs 5' to run 
{}.pt is the name of model you want to train with DOA 
alpha is learning rate of PGD e.g. 0.01
iters is the iterations of PGD e.g.30
out is name of your final model e.g.99
search is method of searching, '0' is exhaustive_search, '1' is gradient_based_search"
epochs is the epoch you want to fine tune your network e.g. 5

Note that ROA is a abstract attacking model simulate the "physical" attacks
Thus there is no restriction for the mask to be rectangle
'''


import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import time
import cv2
import torchfile
import matplotlib.pyplot as plt
from train_model import data_process_lisa
import argparse
import copy
from test_model import test
from train_model import Net
from sticker_attack import ROA

from save_image import save_image





def sticker_train_model(model, criterion, optimizer, scheduler,alpha, iters, search,  num_epochs=10):
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

            for inputs, labels in dataloaders[phase]:
                roa = ROA( model,32) # 32 is the image size 
                
                with torch.set_grad_enabled(search==1):
                    if search == 0:
                        inputs = roa.exhaustive_search(inputs, labels, args.alpha, args.iters, args.width, \
                            args.height, args.stride, args.stride)
                    # if the number is wrong, run gradient_based_search, since this method can save time
                    else:
                        inputs = roa.gradient_based_search(inputs, labels, args.alpha, args.iters, args.width, \
                            args.height, args.stride, args.stride, args.nums_choose)

                    save_image('1112sticker_'+str(args.width)+str(args.iters),inputs)
                    
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
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
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
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
    parser.add_argument("-alpha", type=float, help="alpha leanrning rate")
    parser.add_argument("-iters", type=int, help="iterations of PGD ")
    parser.add_argument("-out", type=int, help="name of final model")
    parser.add_argument("-search", type=int, help="method of searching, \
        '0' is exhaustive_search, '1' is gradient_based_search")
    parser.add_argument("-epochs", type=int, help="epochs")
    parser.add_argument("--stride", type=int, default=2, help="the skip pixels when searching")
    parser.add_argument("--width", type=int, default= 7, help="width of the rectuagluar occlusion")
    parser.add_argument("--height", type=int, default=7, help="height of the rectuagluar occlusion")
    parser.add_argument("--nums_choose", type=int, default=5, help="number of potential positons for final search")
    args = parser.parse_args()

    for i in [5,6,7]:
        seed = i
        torch.manual_seed(seed)
        torch.cuda.empty_cache()
        print('Outout model name is ../donemodel/new_sticker_model0'+str(args.out)+str(seed)+'.pt')
        dataloaders,dataset_sizes =data_process_lisa(batch_size =256) #256
    
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model_ft = Net() 
        model_ft.load_state_dict(torch.load('../donemodel/'+args.model))
        #model_ft.load_weights()
        model_ft.to(device)
    
        # model_ft = nn.DataParallel(model,device_ids=[0,1])
    
        criterion = nn.CrossEntropyLoss()
    
        #optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
        optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.01)
        #optimizer_ft = optim.Adadelta(model_ft.parameters(), lr=0.1, rho=0.9, eps=1e-06, weight_decay=0.01)

        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=100, gamma=0.3)

    
        model_ft = sticker_train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,args.alpha, args.iters,args.search, num_epochs=args.epochs)
        test(model_ft,dataloaders,dataset_sizes)

        torch.save(model_ft.state_dict(), '../donemodel/new_sticker_model0'+str(args.out)+str(seed)+'.pt')



