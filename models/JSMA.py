import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
from new_vgg_face import VGG_16
from save_image import save_image2
import foolbox
import torchvision
import os 
from torchvision import datasets, models, transforms


def data_process(batch_size=64):
    # Data augmentation and normalization for training
    # Just normalization for validation
    #mean = [0.367035294117647,0.41083294117647057,0.5066129411764705]
    data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(size = (224,224)), 
        #transforms.RandomCrop(224),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #do not have transforms before
       # transforms.Normalize(mean, [1/255, 1/255, 1/255])
    ]),
    'val': transforms.Compose([
        transforms.Resize(size = (224,224)),
        #transforms.CenterCrop(224),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
       # transforms.Normalize(mean, [1/255, 1/255, 1/255])
    ]),
    'test': transforms.Compose([
        transforms.Resize(size = (224,224)),
        #transforms.CenterCrop(224),
        transforms.ToTensor(),
        #transforms.Normalize(mean, [1/255, 1/255, 1/255])
    ]),
    }
								
    data_dir = '..'   # change this 
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
    #parser.add_argument("attack", type=str, help="attack")
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")   

    model = VGG_16() 
    model.load_state_dict(torch.load('../donemodel/'+args.model))
    
    
    print("test model is ", args.model)
    model.eval()
    batch_size = 1
    dataloaders,dataset_sizes =data_process(batch_size)

    # model.to(device)
    preprocessing = dict( mean = [0.5066129411764705,0.41083294117647057, 0.367035294117647], std=[1/255, 1/255, 1/255], axis=-3)
    fmodel = foolbox.models.PyTorchModel(
        model.eval(), bounds=(0, 1), num_classes=10,preprocessing=preprocessing)
    
    correct = 0 
    total = 0
    kk = 0 
    torch.manual_seed(12345)
    ten = 0 
    hundred = 0 
    thousand = 0 
    tenthousand = 0 
    
    for max_i in [100000]:
        for data in dataloaders['test']:
            images, labels = data
    
            images = images[:,[2,1,0],:,:]
            X, y = images.cuda().cpu().numpy(), labels.cuda().cpu().numpy()
            for i in range(X.shape[0]):
                metric = foolbox.distances.L0
                attack = foolbox.v1.attacks.SaliencyMapAttack(fmodel, distance=metric)
                
                image, label = X[i], y[i]
                total +=1
                adversarial = attack(image, label, max_iter=max_i,  theta=0.1 ,num_random_targets = 1 )
                
                
                if adversarial is not None:
                    
                    #save_image2("jsmadiff",image - adversarial)
                    #print(np.sum(( (image - adversarial) == 0)*1, 0 ))
                    count =  224*224 - np.sum( (np.sum(( (image - adversarial) == 0)*1, 0 ) == 3)*1 ) 
                    ten += (count <= 10 ) *1
                    hundred += (count <= 100 ) *1
                    thousand += (count <= 1000 ) *1
                    tenthousand += (count <= 10000 ) *1
                    print(count)
                    
                    #print( 150528 - np.sum(( (image - adversarial) == 0)*1 ))
                    
                    #save_image2("jsma",adversarial)
                    
                    #print(total)
                    #yp = np.argmax(fmodel.forward_one(adversarial))
                    #yk = np.argmax(fmodel.forward_one(image))
                    # if yp == label:
                    #     correct += 1
                    kk += 1
                    
     
        print("all", (total-kk)/total, "  10: ", (total-ten)/total ,"  100: ", (total-hundred)/total, "  1000: ",(total-thousand)/total,"  10000: ", (total-tenthousand)/total )
        print('Accuracy of the network on the %s test images: %10.5f %%' % (total,100 * (total-kk)/total))






