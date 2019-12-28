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
from save_image import save_image 
from save_image import save_image2 
from origin_train import data_process


class sticker(object):
    '''
    Make sticker 
    '''

    def __init__(self, base_classifier: torch.nn.Module, alpha, iters):
        self.base_classifier = base_classifier
        self.alpha = alpha
        self.iters = iters

    def predict(self, X, y, width, height, xskip, yskip):
        model = self.base_classifier
        model.eval()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        X = X.to(device)
        y = y.to(device)
        mean = torch.Tensor(np.array([129.1863 , 104.7624,93.5940])).view(1, 3, 1, 1)
        mean = mean.to(device)
        
        max_loss = torch.zeros(y.shape[0]).to(y.device) -100
        all_loss = torch.zeros(y.shape[0]).to(y.device) 
        xtimes = 224//xskip
        ytimes = 224//yskip

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
                #print(all_loss)
        #print(output_j, output_i)
        #save_image("sticker",max_delta)
        
        #print(output_j,output_i)
        zero_loss =  np.transpose(np.argwhere(max_loss.cpu()==0))
        #print(zero_loss)
        for ind in zero_loss:
            output_j[ind] = torch.randint(ytimes,(1,))
            output_i[ind] = torch.randint(xtimes,(1,))
            
        with torch.set_grad_enabled(True):
            return self.cpgd(X,y,width, height, xskip, yskip, output_j, output_i ,mean)
        #print(output_j,output_i)
        #return self.choose_color(X,y,max_loss,width, height, xskip, yskip, output_j, output_i ,mean )
        
    def cpgd(self,X,y,width, height, xskip, yskip, out_j, out_i,mean):
        model = self.base_classifier
        model.eval()
        alpha = self.alpha
        num_iter = self.iters
        sticker = torch.zeros(X.shape, requires_grad=True)
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
        #X1.data = X.detach()
        #print(X1.detach().size(),X.size())
        #sm = torch.nn.Softmax()
        #save_image("00nosticker",(X).detach())
        #save_image("00exhstickerin",(X1).detach())
        #,top10elabel = torch.topk(sm(model(X1)),10)
        #top2prob,top2label = torch.topk(sm(model(X)),2)
        #print("predict ori ",top2prob,top2label)
        #print("predict exh prob",top10eprob)
        #print("predict exh label",top10elabel)
        #delta = torch.zeros_like(X)
        eps = torch.zeros_like(X)
        
        for t in range(num_iter):
            loss = nn.CrossEntropyLoss()(model(X1), y)
            loss.backward()
            
            eps.data = (eps.detach() + 0.5 * X1.grad.detach().sign()).clamp(-2,2)
            
            X1.data = (X.detach() + eps.detach())*(1-sticker) + X1.detach()*sticker
            
            X1.data = (X1.detach() + alpha*X1.grad.detach().sign()*sticker)
            X1.data = ((X1.detach() + mean).clamp(0,255)-mean)
            
            X1.grad.zero_()
        save_image("stickerpgd_1",(X1).detach())
        return (X1).detach()

    def find_gradient(self,X,y,width,height, xskip, yskip):
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
        save_image2("grad",gradient.detach()*255/2 + 255/2)
        mean = torch.Tensor(np.array([129.1863 , 104.7624,93.5940])).view(1, 3, 1, 1)
        mean = mean.to(device)
        xtimes = 224//xskip
        ytimes = 224//yskip
        nums = 30
        output_j1 = torch.zeros(y.shape[0]).repeat(nums).view(y.shape[0],nums)
        output_i1 = torch.zeros(y.shape[0]).repeat(nums).view(y.shape[0],nums)
        matrix = torch.zeros([ytimes*xtimes]).repeat(1,y.shape[0]).view(y.shape[0],ytimes*xtimes)
        #print(output_j1.size(),matrix.size())
        max_loss = torch.zeros(y.shape[0]).to(y.device)
        all_loss = torch.zeros(y.shape[0]).to(y.device)
        
        for i in range(xtimes):
            for j in range(ytimes):
                num = gradient[:,:,yskip*j:(yskip*j+height),xskip*i:(xskip*i+width)]
                loss = torch.sum(torch.sum(torch.sum(torch.mul(num,num),1),1),1)
                matrix[:,j*ytimes+i] = loss
        #print(matrix)
        topk_values, topk_indices = torch.topk(matrix,nums)
        #print(topk_values.size(),topk_indices.size())
        output_j1 = topk_indices//xtimes
        output_i1 = topk_indices %xtimes
        
        #print(output_j,output_i)
        output_j = torch.zeros(y.shape[0]) + output_j1[:,0].float()
        output_i = torch.zeros(y.shape[0]) + output_i1[:,0].float()
        with torch.set_grad_enabled(False):
            for l in range(output_j1.size(1)):
                sticker = X + mean
                for m in range(output_j1.size(0)):
                #print(output_j1)
                    sticker[m,:,yskip*output_j1[m,l]:(yskip*output_j1[m,l]+height),xskip*output_i1[m,l]:(xskip*output_i1[m,l]+width)] = 255/2
                sticker1 = sticker.detach() - mean.detach()
                all_loss = nn.CrossEntropyLoss(reduction='none')(model(sticker1),y)
                save_image('stickerkkk',sticker1)
                #print(sticker1.size())
                padding_j = torch.zeros(y.shape[0]) + output_j1[:,l].float()
                #print(padding_j)
                padding_i = torch.zeros(y.shape[0]) + output_i1[:,l].float()
                output_j[all_loss > max_loss] = padding_j[all_loss > max_loss]
                output_i[all_loss > max_loss] = padding_i[all_loss > max_loss]
                max_loss = torch.max(max_loss, all_loss)
            #print(output_j,output_i)
        return self.cpgd(X,y,width, height, xskip, yskip, output_j, output_i ,mean)
        

    
        

def run(model, stride,width,height,dataloaders):
    #torch.manual_seed(123456)
    print("success1")
    count = 0
    total = 0
    
    for images,labels in dataloaders["test"]:
        
        with torch.set_grad_enabled(True):
            images = images[:,[2,1,0],:,:]
            save_image("sticker_ori"+str(labels.data),images.data)
            total +=  (labels == labels).sum().item()
            stickers = sticker(model,args.alpha,args.iters)
            before_time = time()
            
            #images = stickers.predict(images, labels,width,height,stride,stride)
            images = stickers.find_gradient(images, labels,width,height,stride,stride)
            #images = stickers.feature_map(images, labels,width,height,stride,stride)
            
            outputs = model(images)
            kk, predicted = torch.max(outputs.data, 1)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            #save_image("stickers_att"+str(labels.data)+str(predicted.data),images.data)
            labels = labels.to(device)
            #print(outputs, predicted,labels)
            count += (predicted == labels).sum().item()
            
        
            after_time = time()
            time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))

            print("acc is ", round(count/total,4),"all is ",total)
            print("{}\t{}\t{}".format(round(count/total,4), total, time_elapsed))
            input("Press Enter to continue...")

    print("done with stride of ", stride)
            #print("-----#------#------#---")
    


    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Predict on many examples')
    parser.add_argument("model", type=str, help="test_model")
    parser.add_argument("alpha", type=int, help="alpha")
    parser.add_argument("iters", type=int, help="iters")
    parser.add_argument("--stride", type=int, default=10, help="batch size")
    parser.add_argument("--width", type=int, default= 70, help="batch size")
    parser.add_argument("--height", type=int, default=70, help="batch size")
    args = parser.parse_args()

    torch.manual_seed(123456)

    model = VGG_16() 
    model.load_state_dict(torch.load('../donemodel/'+args.model))

    model.eval()
    batch_size = 1
    dataloaders,dataset_sizes =data_process(batch_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    run(model, args.stride ,args.width ,args.height ,dataloaders )
    


    # def find_gradient1(self,X,y,width,height, xskip, yskip):
    #     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #     gradient = torch.zeros_like(X,requires_grad=True).to(device)
    #     X1 = torch.zeros_like(X,requires_grad=True)
    #     X = X.to(device)
    #     X1.data = X.detach().to(device)
    #     y = y.to(device)
    #     model = self.base_classifier
    #     loss = nn.CrossEntropyLoss()(model(X1), y) + nn.l1loss(X1,)
    #     loss.backward()
    #     gradient.data = X1.grad.detach()
    #     max_val,indice = torch.max(torch.abs(gradient.view(gradient.shape[0], -1)),1)
    #     gradient = gradient /max_val[:,None,None,None]
    #     X1.grad.zero_()
    #     save_image2("grad",gradient.detach()*255/2 + 255/2)
    #     mean = torch.Tensor(np.array([129.1863 , 104.7624,93.5940])).view(1, 3, 1, 1)
    #     mean = mean.to(device)
    #     xtimes = 224//xskip
    #     ytimes = 224//yskip
    #     nums = 30 
    #     output_j = torch.zeros(y.shape[0])
    #     output_i = torch.zeros(y.shape[0])

    #     max_loss = torch.zeros(y.shape[0]).to(y.device)
    #     all_loss = torch.zeros(y.shape[0]).to(y.device)
    #     for i in range(xtimes):
    #         for j in range(ytimes):
    #             num = gradient[:,:,yskip*j:(yskip*j+height),xskip*i:(xskip*i+width)]
    #             all_loss = torch.sum(torch.sum(torch.sum(torch.mul(num,num),1),1),1)
    #             #print(all_loss.shape)
    #             padding_j = torch.zeros(y.shape[0]) + j
    #             padding_i = torch.zeros(y.shape[0]) + i
    #             output_j[all_loss > max_loss] = padding_j[all_loss > max_loss]
    #             output_i[all_loss > max_loss] = padding_i[all_loss > max_loss]
    #             max_loss = torch.max(max_loss, all_loss)
    #     #print(output_j,output_i)
    #     return self.cpgd(X,y,width, height, xskip, yskip, output_j, output_i ,mean)
    

    # def feature_map(self,X,y,width,height, xskip, yskip):
    #     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #     gradient = torch.zeros_like(X,requires_grad=True).to(device)
    #     model = self.base_classifier
    #     X = X.to(device)
    #     y = y.to(device)
    #     feature = model.find_feature(X)
    #     feature = feature.to(device)
        
        
    #     feature = torch.sum(feature, dim=1,keepdim = True)
    #     feature = torch.cat((feature, feature, feature), 1)
    #     max_val,indice = torch.max(torch.abs(feature.view(feature.shape[0], -1)),1)
    #     feature = feature /max_val[:,None,None,None]
    #     print(feature.size())
    #     save_image2("feature",feature.detach()*255/2 + 255/2)



# def choose_color(self,X,y,loss,width, height, xskip, yskip, out_j, out_i,mean):
#         #torch.manual_seed(123456)
#         max_loss = loss
#         max_delta = X.detach().clone()
#         model = self.base_classifier
#         all_color = np.loadtxt('./common_colors.txt', dtype = 'int')
#         all_loss = torch.zeros(y.shape[0]).to(y.device) 
#         #print(all_color)

#         for color in all_color:
#             color_image = X.detach().clone() + mean.detach()
#             for num,ii in enumerate(out_i):
#                 color0 = int(color[0])
#                 color1 = int(color[1])
#                 color2 = int(color[2])
#                 j = int(out_j[num].item())
#                 i = int(ii.item())
#                 color_image[num,0,yskip*j:(yskip*j+height),xskip*i:(xskip*i+width)] = color2
#                 color_image[num,1,yskip*j:(yskip*j+height),xskip*i:(xskip*i+width)] = color1
#                 color_image[num,2,yskip*j:(yskip*j+height),xskip*i:(xskip*i+width)] = color0
#             color_image = color_image.detach() - mean
#             #save_image('sticker'+str(color0),color_image)
#             all_loss = nn.CrossEntropyLoss(reduction='none')(model(color_image),y)
#             # reslogits = model(color_image)
#             # for k in range(reslogits.shape[0]):
#             #     all_loss[k] = max(reslogits[k].data) - reslogits[k][y[k]]
                
#             max_delta[all_loss >= max_loss] = color_image.detach()[all_loss >= max_loss]
#             max_loss = torch.max(max_loss, all_loss)
#         save_image('sticker22',max_delta)
        
#         #outputs = model(max_delta)
#         #kk, predicted = torch.max(outputs.data, 1)
#         #count = (predicted == y).sum().item()
#         #print(count)
#         #print(max_loss)
#         return max_delta

