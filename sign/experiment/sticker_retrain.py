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
from save_image import save_image
from sticker_attack import sticker




def sticker_train_model(model, criterion, optimizer, scheduler, num_epochs=10):
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
                stickers = sticker( model,args.alpha,args.iters)
                
                with torch.set_grad_enabled(args.mode=="grad"):
                    if args.mode == "grad":
                        inputs = stickers.find_gradient(inputs, labels, args.width,args.height,args.stride,args.stride)
                    if args.mode == "exh":
                        inputs = stickers.predict(inputs, labels, args.width,args.height,args.stride,args.stride)
                    save_image('2sticker_'+str(args.width)+str(args.iters)+args.mode,inputs)
                    
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
    
    parser = argparse.ArgumentParser(description='Predict on many examples')
    parser.add_argument("model", type=str, help="ori_model")
    parser.add_argument("alpha", type=float, help="alpha")
    parser.add_argument("iters", type=int, help="iters")
    parser.add_argument("out", type=int, help="out")
    parser.add_argument("mode", type=str, help="mode")
    parser.add_argument("epochs", type=int, help="epochs")
    parser.add_argument("--stride", type=int, default=2, help="stride") #2
    parser.add_argument("--width" , type=int, default=7, help="width")
    parser.add_argument("--height", type=int, default=7, help="height")
    args = parser.parse_args()
    for i in [5,6,7]:
        seed = i
        torch.manual_seed(seed)
        torch.cuda.empty_cache()
        print('../donemodel/new_sticker_model0'+str(args.out)+str(seed)+'.pt')
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

    
        model_ft = sticker_train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,num_epochs=args.epochs)
        test(model_ft,dataloaders,dataset_sizes)

        torch.save(model_ft.state_dict(), '../donemodel/new_sticker_model0'+str(args.out)+str(seed)+'.pt')



