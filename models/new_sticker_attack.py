import torch
import torch.nn as nn
import torch.optim as optim
from origin_train import data_process
import numpy as np
import argparse
import torch.nn.functional as F
# from linf_attack import imshow
import torchvision
import cv2
from torchvision import datasets, models, transforms
import sys
import numpy
from new_vgg_face import VGG_16
from vgg_face import VGG_16_ori
from save_image import save_image 
numpy.set_printoptions(threshold=sys.maxsize)


parser = argparse.ArgumentParser(description='test')
parser.add_argument("model", type=str, help="test_model")
args = parser.parse_args()


def choose_color(model,X,y,glass,mean):
    #torch.manual_seed(1111)
    model.eval()
    potential_starting_color0 = [128,220,160,200,220]
    potential_starting_color1 = [128,130,105,175,210]
    potential_starting_color2 = [128,  0, 55, 30, 50]

    max_loss = torch.zeros(y.shape[0]).to(y.device)
    max_delta = torch.zeros_like(X)
     
    
    for i in range(len(potential_starting_color0)):
        delta1 = torch.zeros(X.size()).to(y.device)
        #imshow(glass,second = 1)

        delta1[:,0,:,:] = glass[0,:,:]*potential_starting_color2[i]
        delta1[:,1,:,:] = glass[1,:,:]*potential_starting_color1[i]
        delta1[:,2,:,:] = glass[2,:,:]*potential_starting_color0[i]

        #print(delta.size())

        all_loss = nn.CrossEntropyLoss(reduction='none')(model(X+delta1-mean),y)
        #save_image('withglass',(X+delta1-mean).detach())
        
        #print(all_loss)
        max_delta[all_loss >= max_loss] = delta1.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    #print(max_loss)
    #print("5 iter over")

    return max_delta



def glass_attack(model, X, y, glass, alpha=1, num_iter=20,momentum=0.4):
    """ Construct glass frame adversarial examples on the examples X"""
    #save_image("inputx",X)
    model.eval()
    mean = torch.Tensor(np.array([129.1863 , 104.7624,93.5940])).view(1, 3, 1, 1)
    de = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mean = mean.to(de)
    # print(mean.size())
    # print((mean*glass).size())
    # print(((X1*(1-glass)).size()) )
    X1 = torch.zeros_like(X,requires_grad = True)
    X1.data = (X+mean)*(1-glass)

    color_glass = choose_color(model,X1,y,glass,mean)
    # imshow(X1+color_glass-mean, title=[ y ])
    X1.data = X1.data+color_glass-mean
    #imshow(X1, title=[ y ])

    delta =torch.zeros_like(X)
    
    for t in range(num_iter):
        
        loss = nn.CrossEntropyLoss()(model(X1), y)
        #print(loss)
        #imshow(X+delta-mean, title=[ y ])
        loss.backward()
        #val, indice = torch.max( torch.abs(delta.grad.detach().view(y.size(0),3*224*224)) ,1)
        #print(val)
        #print(X1.grad)

        delta_change =  X1.grad.detach()*glass
        max_val,indice = torch.max(torch.abs(delta_change.view(delta_change.shape[0], -1)),1)
        #print(max_val)
        r = alpha * delta_change /max_val[:,None,None,None]

        if t == 0:
            delta.data = r
        else:
            delta.data = momentum * delta.detach() + r

        #imshow(X1+delta.detach())
        delta.data[(delta.detach() + X1.detach() + mean) >255] = 0 
        delta.data[(delta.detach() + X1.detach() + mean) < 0 ] = 0 
        X1.data = (X1.detach() + delta.detach())
        #X1.data = (X1.detach() + mean).clamp(0,255) - mean

        X1.data = torch.round(X1.detach()+mean) - mean
        #print(X1.detach())
          
        X1.grad.zero_()

    #print(np.round(delta.detach().cpu().numpy()[0,2,:,:]))
    return (X1).detach()




if __name__ == "__main__":
    
    alpha   = 20
    #itera   = [1 , 2  , 3  , 5  , 7  , 10 , 20 , 50 , 100 , 300 ]
    itera = [30]
    restart = 1
    
    im = cv2.imread('./dataprepare/test1.png')
    im = torch.Tensor(im).permute(2, 0, 1).view(1, 3, 224, 224)
    glass1 = cv2.imread('./dataprepare/silhouette.png')
    glass = transforms.ToTensor()(glass1)
    
    model = VGG_16_ori()
    model.load_weights()
    #torch.manual_seed(12345)
    model.eval()
    im -= torch.Tensor(np.array([129.1863, 104.7624, 93.5940])).view(1, 3, 1, 1)
    y = torch.zeros(1, dtype=torch.long)+477
    print(y)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    im = im.to(device)
    y = y.to(device)
    glass = glass.to(device)
    
    Xadv = glass_attack(model,im,y,glass,alpha=20,num_iter=100)
    preds = F.softmax(model(Xadv),dim = 1)
    save_image("glass_attack_test",Xadv)
    values = preds[0][477]
    print(values)
    val, inc = preds.max(-1)
    print(val,inc)
    
