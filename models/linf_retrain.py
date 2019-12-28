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
from origin_train import data_process
from linf_attack import l_inf_pgd
import argparse
import copy
from origin_test import test
from new_vgg_face import VGG_16
from save_image import save_image




plt.ion()   # interactive mode


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
                inputs = inputs[:,[2,1,0],:,:]
                inputs = inputs.to(device1)
                labels = labels.to(device1)
                delta =  l_inf_pgd(model, inputs, labels, epsilon=args.eps, alpha=args.alpha, num_iter=args.iters,
                         randomize=True) 
                         
                inputs = inputs + delta
                save_image('linf_attack_train'+str(args.eps)+str(args.iters),inputs)
                if phase == 'train':
                    model.train()  # Set model to training mode
                
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
                #print(preds,labels)
                running_corrects += torch.sum(preds == labels.data)
                #print(running_loss,running_corrects)
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
    parser = argparse.ArgumentParser(description='ori_model')
    parser.add_argument("model", type=str, help="ori_model")
    parser.add_argument("-eps", type=int, help="eps")
    parser.add_argument("-alpha", type=int, help="alpha")
    parser.add_argument("-iters", type=int, help="iters")
    parser.add_argument("-out", type=int, help="out")
    parser.add_argument("-epochs", type=int, help="epochs")
    args = parser.parse_args()
    
    torch.manual_seed(123456)
    dataloaders,dataset_sizes =data_process(batch_size =64)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_ft = VGG_16() 
    #model_ft.load_weights()
    model_ft.load_state_dict(torch.load('../donemodel/'+args.model))
    model_ft.to(device)
    
    # model_ft = nn.DataParallel(model,device_ids=[0,1])
    
    print("eps is ",args.eps,"  ","alpha is ", args.alpha,"  ","iteration is ", args.iters,"  ", "out is", args.out,"  ", "epochs is ", args.epochs)
    
    criterion = nn.CrossEntropyLoss()
    
    #optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.0001)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)

    
    model_ft = pgd_train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,num_epochs=args.epochs)
    test(model_ft,dataloaders,dataset_sizes)

    torch.save(model_ft.state_dict(), '../donemodel/new_linf_model0'+ str(args.out) +'.pt')







