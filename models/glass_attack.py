import torch
import torch.nn as nn
import torch.optim as optim
from origin_train import data_process
import numpy as np
import argparse
# from linf_attack import imshow
import torchvision
import cv2
from torchvision import datasets, models, transforms
import sys
import numpy
from new_vgg_face import VGG_16
from save_image import save_image 
numpy.set_printoptions(threshold=sys.maxsize)





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
    
    with torch.set_grad_enabled(False):
        color_glass = choose_color(model,X1,y,glass,mean)
    with torch.set_grad_enabled(True):
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
            #print(X1.grad.detach())
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
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument("model", type=str, help="test_model")
    args = parser.parse_args()
    
    alpha   = 20
    itera   = [1 , 2  , 3  , 5  , 7  , 10 , 20 , 50 , 100 , 300 ]
    #itera = [300]
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
            images = images[:,[2,1,0],:,:]
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

                    
                    # imshow(Xadv, title=[ labels])
                    _, predicted = torch.max(outputs.data, 1)
                
                    #print("predicted is ",predicted,"labels are",labels)
                    check_num += (predicted == labels)
                    #print("yes",check_num)
            correct += (correct_num == check_num).sum().item()
            # if predicted.data != labels.data:
            #     save_image('glass_uattack'+str(labels.data)+'_'+str(total),images.data)
            #     save_image('glass_attack'+str(predicted.data)+'_'+str(total),Xadv.data)
            
            print("one batch is over, batch size", labels.size(0), "correct predict is " ,(correct_num == check_num).sum().item())

        print("alpha is ",alpha,", iteration is ",itera[i]," restart is ", restart)
        print('Accuracy of the network on the %s test images: %10.5f %%' % (total,100 * correct / total))
















# import torch
# import torch.nn as nn
# import torch.optim as optim
# from origin_train import data_process
# import numpy as np
# import argparse
# import cv2
# import torchvision import transforms
# from new_vgg_face import VGG_16

# parser = argparse.ArgumentParser(description='test')
# parser.add_argument("model", type=str, help="test_model")
# args = parser.parse_args()

# def glass_attack(model, X, y, glass,epsilon=1, alpha=1, num_iter=20, randomize=True):
#     """ Construct glass frame adversarial examples on the examples X"""
#     model.eval()
#     mean = torch.Tensor(np.array([129.1863, 104.7624, 93.5940])).view(1, 3, 1, 1)
#     de = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     mean = mean.to(de)
    
#     if randomize:
#         delta = torch.rand_like(X, requires_grad=True)
#         delta.data = delta.data * 2 * epsilon - epsilon
#         delta.data = (delta.data + X + mean).clamp(0,255)-(X+mean)
#     else:
#         delta = torch.zeros_like(X, requires_grad=True)
    
#     for t in range(num_iter):
        
#         loss = nn.CrossEntropyLoss()(model(X + delta), y)
#         loss.backward()
#         # print(torch.max(torch.max(delta.grad.detach())))
#         delta0 = delta.grad.detach()/torch.max(torch.max(delta.grad.detach()))
#         # print(delta0)
#         delta.data = (delta + alpha*delta0).clamp(-epsilon,epsilon)

#         #delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
#         delta.data = ( (delta.data +X + mean).clamp(0,255)-(X+mean) )*glass        
#         delta.grad.zero_()
        
#     return delta.detach()


# def run(num_iter1):
#     model = torch.load('/ris/tongwu/yvorobeychik/Active/donemodel/'+args.model)
#     print("test model is ", args.model)
#     model.eval()
#     dataloaders,dataset_sizes =data_process(batch_size =1)
    
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
#     glass1 = cv2.imread('/home/research/tongwu/glass/models/dataprepare/silhouette.png')
#     glass = transforms.ToTensor()(glass1)
    
#     correct = 0
#     total = 0
#     for data in dataloaders:
#         images, labels = data
#         #print(images)
#         glass = glass.to(device)        
#         images = images.to(device)
#         labels = labels.to(device)
#         count = 0 
#         num = 5
#         total += 1
#         for i in range(num):
#             images = images + glass_attack(model, images, labels, glass, epsilon=255, alpha=20, num_iter=num_iter1, randomize=True)
#             with torch.no_grad():
#                 model.eval()
#                 outputs = model((images).clamp(0,1))
#                 _, predicted = torch.max(outputs.data, 1)
#                 count += (predicted == labels).sum().item()

#         if count == num:
#             correct += 1
#         #print(count,correct/total,total)
#         # if total ==100:
#         #     break
        

#     print("final accuacy", correct/470)

# if __name__ == "__main__":
#     eps     = [255 , 255 , 255 , 255 , 255 , 255 , 255]
#     alpha   = [20  , 20  , 20  , 20  , 20  , 20  , 20 ]
#     itera   = [1   , 2   , 3   , 5   , 7   , 10  , 20 ]
#     restart = [5   , 5   , 5   , 5   , 5   , 5   , 5  ]
#     # first one: 91% (check)
    
#     model = torch.load('../donemodel/'+args.model)
#     print("test model is ", args.model)
    

    
#     model.eval()
#     batch_size = 32
#     dataloaders,dataset_sizes =data_process(batch_size)
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
#     glass1 = cv2.imread('./dataprepare/silhouette.png')
#     glass = transforms.ToTensor()(glass1)
    
    
#     for i in range(len(eps)):
#         correct = 0 
#         total = 0 
#         torch.manual_seed(12345)
#         for data in dataloaders['test']:
#             images, labels = data
#             glass  = glass.to(device)
#             images = images.to(device)
#             labels = labels.to(device)
#             total += labels.size(0)
            
#             check_num   = torch.zeros([1,labels.size(0)], dtype= torch.uint8,device=device)
#             correct_num = torch.zeros([1,labels.size(0)], dtype= torch.uint8,device=device) + restart[i]
#             for j in range(restart[i]):
#                 delta = glass_attack(model, images, glass, labels, eps[i], alpha[i] ,itera[i] ,True)
#                 with torch.no_grad():
#                     model.eval()
#                     outputs = model(images + delta)
#                     _, predicted = torch.max(outputs.data, 1)
                
#                     #print("predicted is ",predicted,"labels are",labels)
#                     check_num += (predicted == labels)
#                     #print("yes",check_num)
#             correct += (correct_num == check_num).sum().item()
#             print("one batch is over, batch size", labels.size(0), "correct predict is " ,(correct_num == check_num).sum().item())

#         print("eps is ",eps[i],", alpha is ",alpha[i],", iteration is ",itera[i]," restart is ", restart[i])
#         print('Accuracy of the network on the %s test images: %10.5f %%' % (total,100 * correct / total))
        
        

