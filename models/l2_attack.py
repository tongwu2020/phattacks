import torch
import torch.nn as nn
import torch.optim as optim
from origin_train import data_process
import numpy as np
import argparse
from new_vgg_face import VGG_16
from save_image import save_image



def norms(Z):
    """Compute norms over all but the first dimension"""
    return Z.view(Z.shape[0], -1).norm(dim=1)[:,None,None,None]


def pgd2(model, X, y, epsilon=0.5, alpha=0.01, num_iter=40, randomize=False, restarts = 1):
    """ Construct l2 adversarial examples on the examples X
    """
    model.eval()
    mean = torch.Tensor(np.array([129.1863, 104.7624, 93.5940])).view(1, 3, 1, 1)
    mean = mean.to(y.device)
    max_loss = torch.zeros(y.shape[0]).to(y.device)
    max_delta = torch.zeros_like(X)
    
    
    for i in range(restarts):
        if randomize:
            delta = torch.rand_like(X, requires_grad=True)
            delta.data = delta.data * 2 * epsilon - epsilon
            delta.data = (delta.data + X + mean).clamp(0,255)-(X+mean)
            
        else:
            delta = torch.zeros_like(X, requires_grad=True)
        
        
        for t in range(num_iter):
            #print("reach")
            loss = nn.CrossEntropyLoss()(model(X + delta ), y)
            loss.backward()
            delta.data += alpha*delta.grad.detach() / norms(delta.grad.detach())
            delta.data = (delta.data +X + mean).clamp(0,255)-(X+mean)
            delta.data *= epsilon / norms(delta.detach()).clamp(min=epsilon)
    
            delta.grad.zero_()
            
        all_loss = nn.CrossEntropyLoss(reduction='none')(model(X+delta),y)
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
        #print(max_delta)

    return max_delta

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument("model", type=str, help="test_model")
    args = parser.parse_args()
    model = VGG_16() 
    model.load_state_dict(torch.load('../donemodel/'+args.model))
    
    print("test model is ", args.model)
    model.eval()
    batch_size = 16
    dataloaders,dataset_sizes =data_process(batch_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    eps     = [0.5  , 1   , 1.5  , 2   , 2.5  , 3  ]
    alpha   = [0.05 , 0.1 , 0.15 , 0.2 , 0.25 , 0.3]
    itera   = [20   , 20  , 20   , 20  , 20   , 20 ]
    restart = [1    , 1   , 1    , 1   , 1    , 1  ]

    for i in range(len(eps)):
        correct = 0 
        total = 0 
        check = 0 
        torch.manual_seed(12345)
        for data in dataloaders['test']:
            images, labels = data
            images = images[:,[2,1,0],:,:]
            images = images.to(device)
            labels = labels.to(device)
            check +=1

            delta = pgd2(model, images, labels, eps[i]*255, alpha[i]*255,itera[i] ,False, restart[i])
            outputs = model(images + delta)
            #save_image("kk",images+delta)
            
            _, predicted = torch.max(outputs.data, 1)
            #total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total += (labels == labels).sum().item()
        
            #print('Accuracy of the network on the %s test images: %10.5f %%' % (total,100 * correct / total))
            #if check % 3 == 0:
                #print("acc ", correct/total, "cor ", correct, "total ", total)
                # print(check,"check")
        print("eps is ",eps[i],", alpha is ",alpha[i],", iteration is ",itera[i]," restart is ", restart[i])
        print('Accuracy of the network on the %s test images: %10.5f %%' % (total,100 * correct / 470))











