# this file is based on https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
# Author: Sasank Chilamkurthy

'''
We use this file to train the clean model to classify the images.
We do not tune a lot of hyperparmeters, all default hyperparameters showed below.
We obtain more than 98% of accuracy
type 'python origin_train.py', note that we do not provide the VGG_FACE.t7, but you can download the model through  
http://www.robots.ox.ac.uk/~vgg/software/vgg_face/src/vgg_face_torch.tar.gz move it to glass/experiment file
'''

from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy 
import torch.nn.functional as F
import torchfile
import numpy as np
from new_vgg_face import VGG_16
#from save_image import save_image 
#uncomment to show images 


# process the data, can use random crop and RandomHorizontalFlip()
def data_process(batch_size=64):
    mean = [0.367035294117647,0.41083294117647057,0.5066129411764705] # the mean value of vggface dataset 
    data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(size = (224,224)), 
        #transforms.RandomCrop(224),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #do not have transforms before
        transforms.Normalize(mean, [1/255, 1/255, 1/255])
    ]),
    'val': transforms.Compose([
        transforms.Resize(size = (224,224)),
        #transforms.CenterCrop(224),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, [1/255, 1/255, 1/255])
    ]),
    'test': transforms.Compose([
        transforms.Resize(size = (224,224)),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, [1/255, 1/255, 1/255])
    ]),
    }
                                
    data_dir = '..'   # change this if the data is in different loaction 
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
                inputs = inputs[:,[2,1,0],:,:] # rgb to bgr
                device1 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                inputs = inputs.to(device1)
                labels = labels.to(device1)
                #print(inputs.size(),labels.size())


                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    
                    outputs = model(inputs)
                    # save_image("ori_img",inputs)
                    # uncomment to see images 
                    
                    
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
    dataloaders,dataset_sizes =data_process(batch_size =64)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = VGG_16()
    model_ft.load_weights()
    #model_ft=nn.DataParallel(model_ft,device_ids=[0,1]) 
    #if you want to use more gpus, other files may change if you use mutliple gpus

    
    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Observe that all parameters are being optimized
    #optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.0001)
    
    # Decay LR by a factor of 0.1 every 10 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)
    
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,num_epochs=30)
    
    print("..............3...............")
    
    torch.save(model_ft.state_dict(), '../donemodel/new_ori_model.pt')  
    
    
