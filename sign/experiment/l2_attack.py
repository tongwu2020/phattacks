import torch
import torch.nn as nn
import torch.optim as optim
from train_model import data_process_lisa
import numpy as np
import argparse
from train_model import Net
from save_image import save_image



def norms(Z):
    """Compute norms over all but the first dimension"""
    return Z.view(Z.shape[0], -1).norm(dim=1)[:,None,None,None]


def pgd2(model, X, y, epsilon=0.5, alpha=0.01, num_iter=40, randomize=False, restarts = 1):
    """ Construct l2 adversarial examples on the examples X
    """
    model.eval()
    max_loss = torch.zeros(y.shape[0]).to(y.device)
    max_delta = torch.zeros_like(X)
    
    
    for i in range(restarts):
        if randomize:
            # delta = torch.rand_like(X, requires_grad=True)
            # delta.data = delta.data * 2 * epsilon - epsilon
            # delta.data = (delta.data + X ).clamp(0,1)-(X)
            
            #delta = torch.rand_like(X, requires_grad=True) 
            #print(delta.data[:,1,1,1])
            #delta.data *= (2.0*delta.data - 1.0)*epsilon 
            #print(delta.data[:,1,1,1])
            #print(epsilon)
            
            #print("norm1 is ",norms(delta.detach()))
            #print("clamp is ",norms(delta.detach()).clamp(min=100) )
            #delta.data /= norms(delta.detach()).clamp(min=epsilon)
            kk = 0 
            for i in range(1):
                delta = (torch.rand_like(X) - 0.5).renorm(p=2, dim=1, maxnorm=epsilon)
                kk += norms(delta.detach())
            print("norm2 is ",delta.detach())
            #print(torch.max(delta.data[:,1,1,1]))
            
        else:
            delta = torch.zeros_like(X, requires_grad=True)
        
        
        for t in range(num_iter):
            #print("reach")
            loss = nn.CrossEntropyLoss()(model(X + delta ), y)
            loss.backward()
            delta.data += alpha*delta.grad.detach() / norms(delta.grad.detach())
            delta.data = (delta.data +X).clamp(0,1)-(X)
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
    model = Net() 
    model.load_state_dict(torch.load('../donemodel/'+args.model))
    
    print("test model is ", args.model)
    model.eval()
    batch_size = 1
    dataloaders,dataset_sizes =data_process_lisa(batch_size)
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
        for data in dataloaders['val']:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            check +=1

            delta = pgd2(model, images, labels, eps[i], alpha[i],itera[i] ,False, restart[i])
            #print("norm3 is ",norms(delta.detach()))
            outputs = model(images + delta)
            save_image("l2attack",images.detach()+delta.detach())
            
            _, predicted = torch.max(outputs.data, 1)
            #total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total += (labels == labels).sum().item()
        
            #print('Accuracy of the network on the %s test images: %10.5f %%' % (total,100 * correct / total))
            # if check % 1 == 0:
            #     print("acc ", correct/total, "cor ", correct, "total ", total)
            #     print(check,"check")
            #     break
        print("eps is ",eps[i],", alpha is ",alpha[i],", iteration is ",itera[i]," restart is ", restart[i])
        print('Accuracy of the network on the %s test images: %10.5f %%' % (total,100 * correct / total))











