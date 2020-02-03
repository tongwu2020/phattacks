# this file is based on https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
# Author: Sasank Chilamkurthy

'''
We use this file to train the clean model to classify the images.
We do not tune a lot of hyperparmeters, all default hyperparameters showed below.
We obtain more than 98% of accuracy
type 'python origin_train.py', to run the model the output model is named new_ori_model.pt

'''


from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from torchvision import models
from torchsummary import summary
import time
import PIL
import os
import copy
import torch.nn.functional as F
import torchfile



def data_process_lisa(batch_size=64):
    data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(size = (32,32)),
        #transforms.RandomResizedCrop(32),
        #transforms.RandomHorizontalFlip(),
        #transforms.ColorJitter(hue=.1, saturation=.1),
        transforms.ToTensor(),
        
    ]),
    'val': transforms.Compose([
        transforms.Resize(size = (32,32)),
        transforms.ToTensor(),
    ]),
    'test': transforms.Compose([
        transforms.Resize(size = (32,32)),
        transforms.ToTensor(),
    ]),
    }
								
    data_dir = '../LISA'   # change this 
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val','test']}


    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                             shuffle=True)
              for x in ['train', 'val','test']}


    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val','test']}
    class_names = image_datasets['train'].classes

    print(class_names)
    print(dataset_sizes)
    return dataloaders,dataset_sizes
    


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 64, 8, stride=2, padding=3)
        self.conv2 = nn.Conv2d(64, 128, 6, stride=2,padding = 0 )
        self.conv3 = nn.Conv2d(128, 128, 5,stride=1, padding = 0)
        self.fc1 = nn.Linear(512, 16)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)


def train_model(model, criterion, optimizer, scheduler, num_epochs=2):
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

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
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
    #torch.manual_seed(123456)
    dataloaders,dataset_sizes =data_process_lisa(batch_size =128)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = Net()

    #model_ft=nn.DataParallel(model_ft,device_ids=[0,1]) 
    #if you want to use the mutliple gpus to run 
    
    model_ft = model_ft.to(device)
    summary(model_ft, (3, 32, 32))
    model_ft.apply(weights_init)
    
    criterion = nn.CrossEntropyLoss()
    
    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.03, momentum=0.9)

    # Decay LR by a factor of 0.1 every 10 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=15, gamma=0.1)
    
    
    
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,num_epochs=30)
    
    print("..............3...............")
    
    torch.save(model_ft.state_dict(), '../donemodel/new_ori_model.pt')
    
    
