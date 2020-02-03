'''
We use this file to test the clean accuracy.
Type 'python origin_test {}.pt '  ({} name of your model want to test) to run
'''

import torch
import torch.nn as nn
import torch.optim as optim
from train_model import data_process_lisa
from train_model import Net
import numpy as np
import argparse
import copy
import torchfile
import torchvision
from torchvision import datasets, models, transforms


def test(model,dataloaders,dataset_sizes):
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    correct = 0 
    total = 0
    #torch.manual_seed(12345)
    with torch.no_grad():
        for data in dataloaders['val']:
            images, labels = data

            # if labels.data == 4: 
            #      continue
            # uncomment to test the adversarial patch 

            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the %s test images: %10.5f %%' % (total,100 * correct / total))


    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument("model", type=str, help="test_model")
    args = parser.parse_args()

    model = Net() 
    model.load_state_dict(torch.load('../donemodel/'+args.model))
    print("test model is ", args.model)
    dataloaders,dataset_sizes =data_process_lisa(batch_size =1)
    
    test(model,dataloaders,dataset_sizes)
    
    