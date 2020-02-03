'''
JSMA attack against models 
Type 'python JSMA.py {}.pt' to run 
{}.pt is the name of model you want to attack by JSMA

Note that the output will be accuracy when changing 10,100,1000,10000 points
The output will be in jsma_output
'''


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
from train_model import Net
#from save_image import save_image2
#uncomment to see images 
import foolbox
import torchvision
import os 
from torchvision import datasets, models, transforms


def data_process(batch_size=64):
    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(size = (32,32)), 
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #do not have transforms before
    ]),
    'val': transforms.Compose([
        transforms.Resize(size = (32,32)),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]),
    'test': transforms.Compose([
        transforms.Resize(size = (32,32)),
        transforms.ToTensor(),
    ]),
    }
                                
    data_dir = '../LISA'   # change this 
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val','test']}


    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                             shuffle=True)
              for x in ['train', 'val','test']}


    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val','test']}
    class_names = image_datasets['train'].classes

    print(class_names)
    print(dataset_sizes)
    return dataloaders,dataset_sizes






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument("model", type=str, help="test_model")
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   

    model = Net() 
    model.load_state_dict(torch.load('../donemodel/'+args.model))
    
    
    print("test model is ", args.model)
    model.eval()
    batch_size = 8
    dataloaders,dataset_sizes =data_process(batch_size)

    # model.to(device)
    preprocessing = dict( mean = [0,0, 0], std=[1, 1, 1], axis=-3)
    fmodel = foolbox.models.PyTorchModel(
        model.eval(), bounds=(0, 1), num_classes=16,preprocessing=preprocessing)
    
    correct = 0 
    total = 0
    kk = 0 
    torch.manual_seed(12345)
    ten = 0 
    hundred = 0 
    thousand = 0 
    tenthousand = 0 
    f = open('./jsma_output.txt', 'w')
    
    for max_i in [100000]:
        for data in dataloaders['val']:
            images, labels = data
    
            X, y = images.cuda().cpu().numpy(), labels.cuda().cpu().numpy()
            k = np.zeros((2000,1))
            
            for i in range(X.shape[0]):
                metric = foolbox.distances.L0
                attack = foolbox.v1.attacks.SaliencyMapAttack(fmodel, distance=metric)
                
                image, label = X[i], y[i]
                total +=1
                adversarial = attack(image, label, max_iter=max_i,  theta=0.1 ,num_random_targets = 1 )
                
                
                if adversarial is not None:
                    
                    #save_image2("jsmadiff",image - adversarial)
                    #uncomment to see some images 
                    count =  32*32 - np.sum( (np.sum(( (image - adversarial) == 0)*1, 0 ) == 3)*1 ) 
                    
                    k[i] = count
                    ten += (count <= 10 ) *1
                    hundred += (count <= 100 ) *1
                    thousand += (count <= 1000 ) *1
                    tenthousand += (count <= 10000 ) *1
                    print(count,file = f, flush=True)
                    #print(count)
                    kk += 1
        f.close()
        print("all", (total-kk)/total, "  10: ", (total-ten)/total ,"  100: ", (total-hundred)/total, "  1000: ",(total-thousand)/total,"  10000: ", (total-tenthousand)/total )
        print('Accuracy of the network on the %s test images: %10.5f %%' % (total,100 * (total-kk)/total))






