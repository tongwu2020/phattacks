'''
This file basically runs Rectangular Occlusion Attacks (ROA) see paper 
Type 'python sticker_attack.py {}.pt -alpha 4 -iters 50 -search 1' to run 
{}.pt is the name of model you want to attack by ROA 
alpha is learning rate of PGD 
iters is the iterations of PGD 
search is method of searching, '0' is exhaustive_search, '1' is gradient_based_search"

Note that ROA is a abstract attacking model simulate the "physical" attacks
Thus there is no restriction for the mask to be rectangle
'''



import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import copy
import cv2
import torch.nn.functional as F
from new_vgg_face import VGG_16
import argparse
from time import time
import datetime
#from save_image import *
#uncomment to see some images 
from origin_train import data_process


class ROA(object):
    '''
    Rectangular Occlusion Attacks class 
    '''

    def __init__(self, base_classifier: torch.nn.Module, alpha, iters):
        self.base_classifier = base_classifier
        self.alpha = alpha
        self.iters = iters

    def exhaustive_search(self, X, y, width, height, xskip, yskip):
        model = self.base_classifier
        model.eval()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        X = X.to(device)
        y = y.to(device)
        mean = torch.Tensor(np.array([129.1863, 104.7624, 93.5940])).view(1, 3, 1, 1)
        mean = mean.to(device)
        
        max_loss = torch.zeros(y.shape[0]).to(y.device) -100
        all_loss = torch.zeros(y.shape[0]).to(y.device) 
        xtimes = (224 - width )//xskip
        ytimes = (224 - height)//yskip

        output_j = torch.zeros(y.shape[0])
        output_i = torch.zeros(y.shape[0])

        for i in range(xtimes):
            for j in range(ytimes):

                sticker = X+ mean
                sticker[:,:,yskip*j:(yskip*j+height),xskip*i:(xskip*i+width)] = 255/2
                sticker1 = sticker.detach() - mean.detach()
                all_loss = nn.CrossEntropyLoss(reduction='none')(model(sticker1),y)
                padding_j = torch.zeros(y.shape[0]) + j
                padding_i = torch.zeros(y.shape[0]) + i
                output_j[all_loss > max_loss] = padding_j[all_loss > max_loss]
                output_i[all_loss > max_loss] = padding_i[all_loss > max_loss]
                max_loss = torch.max(max_loss, all_loss)

        # when the max loss is zero, we cannot choose one part to attack, we will randomly choose a positon 
        zero_loss =  np.transpose(np.argwhere(max_loss.cpu()==0))

        for ind in zero_loss:
            output_j[ind] = torch.randint(ytimes,(1,))
            output_i[ind] = torch.randint(xtimes,(1,))
            
        with torch.set_grad_enabled(True):
            return self.cpgd(X,y,width, height, xskip, yskip, output_j, output_i ,mean)


    def gradient_based_search(self,X,y,width,height, xskip, yskip):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        gradient = torch.zeros_like(X,requires_grad=True).to(device)
        X1 = torch.zeros_like(X,requires_grad=True)
        X = X.to(device)
        X1.data = X.detach().to(device)
        y = y.to(device)
        model = self.base_classifier
        loss = nn.CrossEntropyLoss()(model(X1), y) 
        loss.backward()
        gradient.data = X1.grad.detach()
        max_val,indice = torch.max(torch.abs(gradient.view(gradient.shape[0], -1)),1)
        gradient = gradient /max_val[:,None,None,None]
        X1.grad.zero_()
        mean = torch.Tensor(np.array([129.1863 , 104.7624,93.5940])).view(1, 3, 1, 1)
        mean = mean.to(device)
        xtimes = 224//xskip
        ytimes = 224//yskip
        nums = 30  #default number of 
        output_j1 = torch.zeros(y.shape[0]).repeat(nums).view(y.shape[0],nums)
        output_i1 = torch.zeros(y.shape[0]).repeat(nums).view(y.shape[0],nums)
        matrix = torch.zeros([ytimes*xtimes]).repeat(1,y.shape[0]).view(y.shape[0],ytimes*xtimes)
        max_loss = torch.zeros(y.shape[0]).to(y.device)
        all_loss = torch.zeros(y.shape[0]).to(y.device)
        
        for i in range(xtimes):
            for j in range(ytimes):
                num = gradient[:,:,yskip*j:(yskip*j+height),xskip*i:(xskip*i+width)]
                loss = torch.sum(torch.sum(torch.sum(torch.mul(num,num),1),1),1)
                matrix[:,j*ytimes+i] = loss

        topk_values, topk_indices = torch.topk(matrix,nums)
        output_j1 = topk_indices//xtimes
        output_i1 = topk_indices %xtimes
        
        output_j = torch.zeros(y.shape[0]) + output_j1[:,0].float()
        output_i = torch.zeros(y.shape[0]) + output_i1[:,0].float()
        with torch.set_grad_enabled(False):
            for l in range(output_j1.size(1)):
                sticker = X + mean
                for m in range(output_j1.size(0)):
                    sticker[m,:,yskip*output_j1[m,l]:(yskip*output_j1[m,l]+height),xskip*output_i1[m,l]:(xskip*output_i1[m,l]+width)] = 255/2
                sticker1 = sticker.detach() - mean.detach()
                all_loss = nn.CrossEntropyLoss(reduction='none')(model(sticker1),y)
                padding_j = torch.zeros(y.shape[0]) + output_j1[:,l].float()
                padding_i = torch.zeros(y.shape[0]) + output_i1[:,l].float()
                output_j[all_loss > max_loss] = padding_j[all_loss > max_loss]
                output_i[all_loss > max_loss] = padding_i[all_loss > max_loss]
                max_loss = torch.max(max_loss, all_loss)
            #print(output_j,output_i)
        return self.cpgd(X,y,width, height, xskip, yskip, output_j, output_i ,mean)


    def cpgd(self,X,y,width, height, xskip, yskip, out_j, out_i,mean):
        model = self.base_classifier
        model.eval()
        alpha = self.alpha
        num_iter = self.iters
        sticker = torch.zeros(X.shape, requires_grad=False)
        for num,ii in enumerate(out_i):
            j = int(out_j[num].item())
            i = int(ii.item())
            sticker[num,:,yskip*j:(yskip*j+height),xskip*i:(xskip*i+width)] = 1
        sticker = sticker.to(y.device)
        
        
        delta = torch.zeros_like(X, requires_grad=True)+255/2  
        #delta = torch.rand_like(X, requires_grad=True).to(y.device)
        #delta.data = delta.data * 255 

        X1 = torch.rand_like(X, requires_grad=True).to(y.device)
        X1.data = X.detach()*(1-sticker)+((delta.detach()-mean)*sticker)

        
        for t in range(num_iter):
            loss = nn.CrossEntropyLoss()(model(X1), y)
            loss.backward()
   
            X1.data = (X1.detach() + alpha*X1.grad.detach().sign()*sticker)
            X1.data = ((X1.detach() + mean).clamp(0,255)-mean)
            X1.grad.zero_()
        return (X1).detach()
        
        

def run(model, stride,width,height,dataloaders):
    torch.manual_seed(123456)
    count = 0
    total = 0
    
    for images,labels in dataloaders["test"]:
        
        with torch.set_grad_enabled(args.search != 0):
            images = images[:,[2,1,0],:,:] # rgb to bgr
            #save_image("ROA_original_image"+str(labels.data),images.data)
            #uncomment to see images 
            total +=  (labels == labels).sum().item()
            ROA_module = ROA(model,args.alpha,args.iters)
            before_time = time()

            if args.search == 0:
                #print("runing exhaustive_search")
                ROA_images = ROA_module.exhaustive_search(images, labels,width,height,stride,stride)
            # if the number is wrong, run gradient_based_search, since this method can save time
            else:
                #print("runing gradient_based_search")
                ROA_images = ROA_module.gradient_based_search(images, labels,width,height,stride,stride)


            outputs = model(ROA_images)
            kk, predicted = torch.max(outputs.data, 1)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            #save_image("ROA_noised_image"+str(labels.data)+str(predicted.data),ROA_images.data)
            #uncomment to see images 
            labels = labels.to(device)
            count += (predicted == labels).sum().item()
        
            after_time = time()
            time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
            print("acc is ", round(count/total,4),"all is ",total)
            print("{}\t{}\t{}".format(round(count/total,4), total, time_elapsed))


    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Predict on many examples')
    parser.add_argument("model", type=str, help="test_model")
    parser.add_argument("-alpha", type=int, help="alpha leanrning rate")
    parser.add_argument("-iters", type=int, help="iterations of PGD ")
    parser.add_argument("-search", type=int, help="method of searching, \
        '0' is exhaustive_search, '1' is gradient_based_search")
    parser.add_argument("--stride", type=int, default=10, help="the skip pixels when searching")
    parser.add_argument("--width", type=int, default= 70, help="width of the rectuagluar occlusion")
    parser.add_argument("--height", type=int, default=70, help="height of the rectuagluar occlusion")
    args = parser.parse_args()

    torch.manual_seed(123456)
    model = VGG_16() 
    model.load_state_dict(torch.load('../donemodel/'+args.model))
    model.eval()
    batch_size = 8
    dataloaders,dataset_sizes =data_process(batch_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    run(model, args.stride ,args.width ,args.height ,dataloaders )
    
