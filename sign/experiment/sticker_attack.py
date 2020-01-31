import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import copy
import cv2
import torch.nn.functional as F
import argparse
from time import time
import datetime
from save_image import save_image 
from train_model import data_process_lisa
from train_model import Net

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os


class ROA(object):
    '''
    Make sticker 
    '''

    def __init__(self, base_classifier: torch.nn.Module, size):
        self.base_classifier = base_classifier
        self.img_size = size 

    def exhaustive_search(self, X, y, alpha, num_iter, width, height, xskip, yskip,random = False):
        
        with torch.set_grad_enabled(False):
    
            model = self.base_classifier
            size = self.img_size
    
            model.eval() 
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            X = X.to(device)
            y = y.to(device)
            
            max_loss = torch.zeros(y.shape[0]).to(y.device) - 100
            all_loss = torch.zeros(y.shape[0]).to(y.device) 
    
            xtimes = (size-width) //xskip
            ytimes = (size-height)//yskip
    
            output_j = torch.zeros(y.shape[0])
            output_i = torch.zeros(y.shape[0])
            
            count = torch.zeros(y.shape[0])
            ones = torch.ones(y.shape[0])
    
            for i in range(xtimes):
                for j in range(ytimes):
                    sticker = X.clone()
                    sticker[:,:,yskip*j:(yskip*j+height),xskip*i:(xskip*i+width)] = 1/2          
                    all_loss = nn.CrossEntropyLoss(reduction='none')(model(sticker),y)
                    padding_j = torch.zeros(y.shape[0]) + j
                    padding_i = torch.zeros(y.shape[0]) + i
                    output_j[all_loss > max_loss] = padding_j[all_loss > max_loss]
                    output_i[all_loss > max_loss] = padding_i[all_loss > max_loss]
                    count +=  (all_loss == max_loss).type(torch.FloatTensor)
                    max_loss = torch.max(max_loss, all_loss)
    
            same_loss = np.transpose(np.argwhere(count>=xtimes*ytimes*0.9))
            for ind in same_loss:
                output_j[ind] = torch.randint(ytimes,(1,))
                output_i[ind] = torch.randint(xtimes,(1,))      
    
            zero_loss =  np.transpose(np.argwhere(max_loss.cpu()==0))
            for ind in zero_loss:
                output_j[ind] = torch.randint(ytimes,(1,))
                output_i[ind] = torch.randint(xtimes,(1,))

        
        with torch.set_grad_enabled(True):
            return self.inside_pgd(X,y,width, height,alpha, num_iter, xskip, yskip, output_j, output_i )



    def gradient_based_search(self, X, y, alpha, num_iter, width, height, xskip, yskip, potential_nums,random = False):

        model = self.base_classifier
        size = self.img_size

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        gradient = torch.zeros_like(X,requires_grad=True).to(device)
        X1 = torch.zeros_like(X,requires_grad=True)
        X = X.to(device)
        y = y.to(device)
        X1.data = X.detach().to(device)
        
        loss = nn.CrossEntropyLoss()(model(X1), y) 
        loss.backward()

        gradient.data = X1.grad.detach()
        max_val,indice = torch.max(torch.abs(gradient.view(gradient.shape[0], -1)),1)
        gradient = gradient /max_val[:,None,None,None]
        X1.grad.zero_()

        xtimes = (size-width) //xskip
        ytimes = (size-height)//yskip
        print(xtimes,ytimes)


        nums = potential_nums
        output_j1 = torch.zeros(y.shape[0]).repeat(nums).view(y.shape[0],nums)
        output_i1 = torch.zeros(y.shape[0]).repeat(nums).view(y.shape[0],nums)
        matrix = torch.zeros([ytimes*xtimes]).repeat(1,y.shape[0]).view(y.shape[0],ytimes*xtimes)
        max_loss = torch.zeros(y.shape[0]).to(y.device)
        all_loss = torch.zeros(y.shape[0]).to(y.device)
        
        for i in range(xtimes):
            for j in range(ytimes):
                num = gradient[:,:,yskip*j:(yskip*j+height),xskip*i:(xskip*i+width)]
                loss = torch.sum(torch.sum(torch.sum(torch.mul(num,num),1),1),1)
                #print(j*xtimes+i)
                matrix[:,j*xtimes+i] = loss
        topk_values, topk_indices = torch.topk(matrix,nums)
        output_j1 = topk_indices//xtimes
        output_i1 = topk_indices %xtimes
        
        output_j = torch.zeros(y.shape[0]) + output_j1[:,0].float()
        output_i = torch.zeros(y.shape[0]) + output_i1[:,0].float()

        with torch.set_grad_enabled(False):
            for l in range(output_j1.size(1)):
                sticker = X.clone()
                for m in range(output_j1.size(0)):
                    sticker[m,:,yskip*output_j1[m,l]:(yskip*output_j1[m,l]+height),xskip*output_i1[m,l]:(xskip*output_i1[m,l]+width)] = 1/2
                sticker1 = sticker.detach()
                all_loss = nn.CrossEntropyLoss(reduction='none')(model(sticker1),y)
                padding_j = torch.zeros(y.shape[0]) + output_j1[:,l].float()
                padding_i = torch.zeros(y.shape[0]) + output_i1[:,l].float()
                output_j[all_loss > max_loss] = padding_j[all_loss > max_loss]
                output_i[all_loss > max_loss] = padding_i[all_loss > max_loss]
                max_loss = torch.max(max_loss, all_loss)
            
        return self.inside_pgd(X,y,width, height,alpha, num_iter, xskip, yskip, output_j, output_i)



       
    def inside_pgd(self, X, y, width, height, alpha, num_iter, xskip, yskip, out_j, out_i, random = False):
        model = self.base_classifier
        model.eval()
        sticker = torch.zeros(X.shape, requires_grad=True)
        for num,ii in enumerate(out_i):
            j = int(out_j[num].item())
            i = int(ii.item())
            sticker[num,:,yskip*j:(yskip*j+height),xskip*i:(xskip*i+width)] = 1
        sticker = sticker.to(y.device)


        if random == False:
            delta = torch.zeros_like(X, requires_grad=True)+1/2  
        else:
            delta = torch.rand_like(X, requires_grad=True).to(y.device)
            delta.data = delta.data * 255


        X1 = torch.rand_like(X, requires_grad=True).to(y.device)
        X1.data = X.detach()*(1-sticker)+((delta.detach())*sticker)
        
        for t in range(num_iter):
            loss = nn.CrossEntropyLoss()(model(X1), y)
            loss.backward()
            X1.data = (X1.detach() + alpha*X1.grad.detach().sign()*sticker)
            X1.data = (X1.detach() ).clamp(0,1)
            X1.grad.zero_()
        save_image("stickerpgd",(X1).detach())
        return (X1).detach()

        
        
        

def run(model, stride, width, height, dataloaders):

    count = 0
    total = 0
    
    for images,labels in dataloaders["val"]:
        with torch.set_grad_enabled(True):
            save_image('uattack'+str(labels),images.data)
            
            total +=  (labels == labels).sum().item()
            ROA_module = ROA(model,32)
            before_time = time()
            
            #images = ROA_module.gradient_based_search(images, labels, 0.1, 5, width, height,stride,stride, 10)
            images = ROA_module.exhaustive_search(images, labels, 0.1, 5, width, height,stride,stride)
            
            save_image('attack'+str(labels),images.data)
            
            outputs = model(images)
            kk, predicted = torch.max(outputs.data, 1)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            labels = labels.to(device)
            count += (predicted == labels).sum().item()
            
        
            after_time = time()
            time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))

            print("acc is ", round(count/total,4),"all is ",total)
            print("{}\t{}\t{}".format(round(count/total,4), total, time_elapsed))
            #input("Press Enter to continue...")

    print("done with stride of ", stride)
            #print("-----#------#------#---")
    



    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict on many examples')
    parser.add_argument("model", type=str, help="test_model")
    parser.add_argument("iters", type=int, help="iters")
    parser.add_argument("--stride", type=int, default=2, help="batch size")
    parser.add_argument("--width", type=int, default= 10, help="batch size")
    parser.add_argument("--height", type=int, default=5, help="batch size")
    args = parser.parse_args()

    torch.manual_seed(123456)

    model = Net() 
    model.load_state_dict(torch.load('../donemodel/'+args.model))

    model.eval()
    batch_size = 1
    dataloaders,dataset_sizes =data_process_lisa(batch_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
   
    run(model, args.stride ,args.width ,args.height ,dataloaders )




