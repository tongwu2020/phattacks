# This file is based on (pytorch version of) https://github.com/evtimovi/robust_physical_perturbations
# It carries its own MIT License.


'''
Sticker Attacks
Type 'python physical_attack.py {}.pt'  to run 
{} is the name of your model want to attack. Note that you cannot attack randomized smoothing in this file, 
please use smooth_glassattack.py
iterations   = [10,100,1000] # this is default numbers we used in experiment, 
which is the iterations of attacks 

Note that the attack is in digit space (not involved rotation and scale) (fixed eyeglass frame mask),
and untargeted (maximize the loss of (f(x),y) )
sticker attack: mask_l1loss_uniform_rectangles.png
other attacks: maskdone4.jpg maskdone5.jpg maskdone6.jpg   
'''

import torch
from torch import Tensor
import torch.nn as nn
import torch.autograd as autograd 
import torch.optim as optim
from train_model import data_process_lisa
from train_model import Net
import numpy as np
import argparse
import copy
import torchfile
import torchvision
from torchvision import datasets, models, transforms
#from save_image import save_image 
#uncomment to see some images 
import cv2




def untarget_attack(model, X, y, lamb, num_iter=100):
    """ Construct target_attack adversarial examples on the examples X"""

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    sticker = cv2.imread('../mask/mask_l1loss_uniform_rectangles.png')
    # maskdone4.jpg maskdone5.jpg maskdone6.jpg   mask_l1loss_uniform_rectangles.png
    sticker = transforms.ToTensor()(sticker)
    sticker = sticker.to(device)
    noise = torch.rand_like(X).float() * sticker
    noiseVar = (noise + X).clamp(0,1) -X
    
    
    X = X.to(device)
    y = y.to(device)
    
    noiseVar = noiseVar.to(device)
    

    model.eval()

    for t in range(num_iter):
        noiseVar = autograd.Variable(noiseVar, requires_grad=True)
        optimizer = optim.Adam([noiseVar], lr=0.1,betas=(0.9, 0.999), eps=1e-08) 
        
        optimizer.zero_grad()
        loss1 = -nn.CrossEntropyLoss()(model(X+noiseVar), y) 
        loss2 = +lamb * torch.norm(noiseVar, 2)
        loss = loss1 + loss2
        #may omit regloss since sticker attack does not need it
        
        loss1.backward(retain_graph=True)
        optimizer.step()
        noiseVar = noiseVar * sticker
        noiseVar = (X+noiseVar).clamp(0,1) -X
        
    #save_image('phy_att_image',X.detach()+noiseVar.detach())
    #uncomment to see some images 
    return X.detach()+noiseVar.detach()
    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument("model", type=str, help="test_model")
    args = parser.parse_args()
    
    model = Net() 
    model.load_state_dict(torch.load('../donemodel/'+args.model))
    print("test model is ", args.model)
    dataloaders,dataset_sizes =data_process_lisa(batch_size =1148)
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    
    
    torch.manual_seed(12345)
    for num in [10,100,1000]:
        correct = 0 
        total = 0
        for data in dataloaders['val']:
            images, labels = data
            #save_image('phy_att_start',X1.detach())
            #uncomment to see some images
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            images = images.to(device)
            images = untarget_attack(model, images, labels, 0.02, num_iter=num)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(predicted.device)).sum().item()
        print('num of iteration is ', num)
        print('Accuracy of the network on the %s test images: %10.5f %%' % (total,100 * correct / total))

    
    

