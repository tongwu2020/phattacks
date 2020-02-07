# This file basically runs Rectangular Occlusion Attacks (ROA) see paper 

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

    def __init__(self, base_classifier, size):
        self.base_classifier = base_classifier
        self.img_size = size 
        """
        :param base_classifier: maps from [batch x channel x height x width] to [batch x num_classes]
        :param size: the image size 
        """

    def exhaustive_search(self, X, y, alpha, num_iter, width, height, xskip, yskip,random = False):
        """
        :param X: images from the pytorch dataloaders
        :param y: labels from the pytorch dataloaders
        :param alpha: the learning rate of inside PGD attacks 
        :param num_iter: the number of iterations of inside PGD attacks 
        :param width: the width of ROA 
        :param height: the height of ROA 
        :param xskip: the skip (stride) when searching in x axis 
        :param yskip: the skip (stride) when searching in y axis 
        :param random: the initialization the ROA before inside PGD attacks, 
                       True is random initialization, False is 0.5 initialization
        """
        
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
        """
        :param X: images from the pytorch dataloaders
        :param y: labels from the pytorch dataloaders
        :param alpha: the learning rate of inside PGD attacks 
        :param num_iter: the number of iterations of inside PGD attacks 
        :param width: the width of ROA 
        :param height: the height of ROA 
        :param xskip: the skip (stride) when searching in x axis 
        :param yskip: the skip (stride) when searching in y axis 
        :param potential_nums: the number of keeping potential candidate position
        :param random: the initialization the ROA before inside PGD attacks, 
                       True is random initialization, False is 0.5 initialization
        """

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
        return (X1).detach()

        


