# this file is based on https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
# Author: Sasank Chilamkurthy

'''
We use this file to adversarial training or curriculum adversarial training the model 
Type 'python linf_retrain.py {}.pt  '  
{} name of your model want to retrain, if doing adversarial training, fill in anything you want to run

Currently, the model is used to adversarial training. For curriculum adversarial training, 
change the code in if __name__ == "__main__": refer to roughly line 117. 
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
from linf_attack import l_inf_pgd
import argparse
import copy
from test_model import test
from train_model import Net
from train_model import data_process_lisa
from train_model import weights_init

parser = argparse.ArgumentParser(description='ori_model')
parser.add_argument("model", type=str, help="ori_model")
args = parser.parse_args()


def pgd_train_model(model, criterion, optimizer, scheduler, num_epochs=10):
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
                device1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                inputs = inputs.to(device1)
                labels = labels.to(device1)
                model1 = model
                
                delta =  l_inf_pgd(model1, inputs, labels, epsilon=8/255, alpha=2/255, num_iter=50,
                         randomize=True) 
                # default hypermeter, can be changed 
                         
                inputs = inputs + delta
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)

                    _, preds = torch.max(outputs, 1)

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
    torch.manual_seed(123456)
    dataloaders,dataset_sizes =data_process_lisa(batch_size =128)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = Net() 
    model_ft.apply(weights_init)
    #model_ft.load_state_dict(torch.load('../donemodel/'+args.model))
    model_ft.to(device)
    
    # model_ft = nn.DataParallel(model,device_ids=[0,1])
    # use multiple gpus
    
    criterion = nn.CrossEntropyLoss()
    
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.01)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)

    
    model_ft = pgd_train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,num_epochs=30)
    test(model_ft,dataloaders,dataset_sizes)

    torch.save(model_ft.state_dict(), '../donemodel/new_linf_model050.pt') # output model







