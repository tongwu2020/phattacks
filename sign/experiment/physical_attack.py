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
from save_image import save_image
import cv2



'''

flags.DEFINE_float('attack_lambda', 0.02, 'The lambda parameter in the attack optimization problem')
flags.DEFINE_float('optimization_rate', 0.1, 'The optimization rate (learning rate) for the Adam optimizer of the attack objective')

flags.DEFINE_string('regloss', '', 'Specifies the regularization loss to use. Options: l1. Anything else defaults to l2')

flags.DEFINE_float('adam_beta1', 0.9, 'The beta1 parameter for the Adam optimizer of the attack objective')
flags.DEFINE_float('adam_beta2', 0.999, 'The beta2 parameter for the Adam optimizer of the attack objective')
flags.DEFINE_float('adam_epsilon', 1e-08, 'The epsilon parameter for the Adam optimizer of the attack objective')

'''


def untarget_attack(model, X, y, lamb, num_iter=100):
    """ Construct target_attack adversarial examples on the examples X"""
 
 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    sticker = cv2.imread('../mask/maskdone4.jpg')
    #maskdone4.jpg   mask_l1loss_uniform_rectangles.png
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
        #print(loss1,loss2)
        #may omit regloss since sticker attack does not need it
        
        loss1.backward(retain_graph=True)
        optimizer.step()
        noiseVar = noiseVar * sticker
        noiseVar = (X+noiseVar).clamp(0,1) -X
        
    save_image('phy_att_l2_image',X.detach()+noiseVar.detach())
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
            #noise = autograd.Variable(modifier, requires_grad=True)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            images = images.to(device)
            images = untarget_attack(model, images, labels, 0.02, num_iter=num)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            #print(predicted)
            correct += (predicted == labels.to(predicted.device)).sum().item()
        print('num of iteration is ', num)
        print('Accuracy of the network on the %s test images: %10.5f %%' % (total,100 * correct / total))

    
    

