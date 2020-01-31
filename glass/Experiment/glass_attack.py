# This file is based on (pytorch version of) https://github.com/mahmoods01/accessorize-to-a-crime 
# Mahmood Sharif


'''
Eyeglass Frame Attack 
Type 'python linf_retrain.py {}.pt'  to run 
{} is the name of your model want to attack. Note that you cannot attack randomized smoothing in this file, 
please use smooth_glassattack.py
itera   = [1 , 2  , 3  , 5  , 7  , 10 , 20 , 50 , 100 , 300 ] # this is default numbers we used in experiment, 
which is the iterations of attacks 

Note that the attack is in digit space (not involved rotation and scale) (fixed eyeglass frame mask),
and untargeted (maximize the loss of (f(x),y) )
'''

import torch
import torch.nn as nn
import torch.optim as optim
from origin_train import data_process
import numpy as np
import argparse
import torchvision
import cv2
from torchvision import datasets, models, transforms
import sys
import numpy
from new_vgg_face import VGG_16
#from save_image import save_image 
#uncomment to see some images 
numpy.set_printoptions(threshold=sys.maxsize)





def choose_color(model,X,y,glass,mean):
    model.eval()
    potential_starting_color0 = [128,220,160,200,220]
    potential_starting_color1 = [128,130,105,175,210]
    potential_starting_color2 = [128,  0, 55, 30, 50]

    max_loss = torch.zeros(y.shape[0]).to(y.device)
    max_delta = torch.zeros_like(X)
     
    
    for i in range(len(potential_starting_color0)):
        delta1 = torch.zeros(X.size()).to(y.device)


        delta1[:,0,:,:] = glass[0,:,:]*potential_starting_color2[i]
        delta1[:,1,:,:] = glass[1,:,:]*potential_starting_color1[i]
        delta1[:,2,:,:] = glass[2,:,:]*potential_starting_color0[i]

        all_loss = nn.CrossEntropyLoss(reduction='none')(model(X+delta1-mean),y)
        max_delta[all_loss >= max_loss] = delta1.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)

    return max_delta


def glass_attack(model, X, y, glass, alpha=1, num_iter=20,momentum=0.4):
    """ Construct glass frame adversarial examples on the examples X"""

    model.eval()
    mean = torch.Tensor(np.array([129.1863 , 104.7624,93.5940])).view(1, 3, 1, 1)
    de = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mean = mean.to(de)
    X1 = torch.zeros_like(X,requires_grad = True)
    X1.data = (X+mean)*(1-glass)
    
    with torch.set_grad_enabled(False):
        color_glass = choose_color(model,X1,y,glass,mean)
    with torch.set_grad_enabled(True):
        X1.data = X1.data+color_glass-mean

        delta =torch.zeros_like(X)
    
        for t in range(num_iter):
        
            loss = nn.CrossEntropyLoss()(model(X1), y)
            loss.backward()

            delta_change =  X1.grad.detach()*glass
            max_val,indice = torch.max(torch.abs(delta_change.view(delta_change.shape[0], -1)),1)
            r = alpha * delta_change /max_val[:,None,None,None]

            if t == 0:
                delta.data = r
            else:
                delta.data = momentum * delta.detach() + r

            delta.data[(delta.detach() + X1.detach() + mean) >255] = 0 
            delta.data[(delta.detach() + X1.detach() + mean) < 0 ] = 0 
            X1.data = (X1.detach() + delta.detach())

            X1.data = torch.round(X1.detach()+mean) - mean
          
            X1.grad.zero_()

        return (X1).detach()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument("model", type=str, help="test_model")
    args = parser.parse_args()
    
    alpha   = 20
    itera   = [1 , 2  , 3  , 5  , 7  , 10 , 20 , 50 , 100 , 300 ] # this is default numbers we used in experiment
    restart = 1

    glass1 = cv2.imread('./dataprepare/silhouette.png')
    glass = transforms.ToTensor()(glass1)
    #print("dd",glass.size())
    
    model = VGG_16() 
    model.load_state_dict(torch.load('../donemodel/'+args.model))

    model.eval()
    batch_size = 64
    dataloaders,dataset_sizes =data_process(batch_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    for i in range(len(itera)):
        correct = 0 
        total = 0 
        torch.manual_seed(12345)
        for data in dataloaders['test']:
            images, labels = data
            images = images[:,[2,1,0],:,:] #rgb to bgr
            glass = glass.to(device)
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)
            
            
            check_num   = torch.zeros([1,labels.size(0)], dtype= torch.uint8,device=device)
            correct_num = torch.zeros([1,labels.size(0)], dtype= torch.uint8,device=device) + restart
            for j in range(restart):
                Xadv = glass_attack(model, images, labels,glass, alpha ,itera[i])
                with torch.no_grad():
                    model.eval()

                    outputs = model(Xadv)

                    _, predicted = torch.max(outputs.data, 1)
                    check_num += (predicted == labels)
            correct += (correct_num == check_num).sum().item()

            # if predicted.data != labels.data:
            #     save_image('glass_uattack'+str(labels.data)+'_'+str(total),images.data)
            #     save_image('glass_attack'+str(predicted.data)+'_'+str(total),Xadv.data)
            # uncomment to see some images
            
            #print("one batch is over, batch size", labels.size(0), "correct predict is " ,(correct_num == check_num).sum().item())

        print("alpha is ",alpha,", iteration is ",itera[i]," restart is ", restart)
        print('Accuracy of the network on the %s test images: %10.5f %%' % (total,100 * correct / total))






