# this file is based on https://github.com/locuslab/smoothing
# This is done by Jeremy Cohen, Elan Rosenfeld, and Zico Kolter 

'''
This file is used to test the randomized smoothing against sticker attacks
Type 'python smooth_glassattack.py {}.pt -sigma 1 '
{}.pt is the name of gaussian model you need to train by gaussian_train.py
1 is sigma of gaussian noise (I use same sigma with the sigma training the gaussian model)


[ 10,100,1000 ] # this is default numbers we used in experiment, 
which is the iterations of attacks 
'''


import argparse
from core import Smooth
from time import time
import torch
import datetime
import torch
from train_model import data_process_lisa
import numpy as np
import torchvision
from train_model import Net
from torchvision import datasets, models, transforms

import os
import copy
import cv2
from physical_attack import untarget_attack




if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Predict on many examples')
    parser.add_argument("model", type=str, help="test_model")
    parser.add_argument("-sigma", type=float, help="noise hyperparameter")
    parser.add_argument("--batch", type=int, default=1000, help="batch size")
    parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
    parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
    parser.add_argument("--N", type=int, default=1000, help="number of samples to use")
    parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 
    print("idx\tlabel\tpredict\tcorrect\ttime")
    batch_size = 1
    dataloaders,dataset_sizes =data_process_lisa(batch_size)
    
    model = Net() 
    model.load_state_dict(torch.load('../donemodel/'+args.model))
    model.to(device)
    smoothed_classifier = Smooth(model, 16, args.sigma)
    torch.manual_seed(123456)

    for i in [ 10,100,1000 ]:
        cor = 0
        tot = 0 
        for k in dataloaders['test']:
            (x, label) = k
            x = x.to(device)
            labels = label.to(device)
    
            before_time = time()
            x1 = untarget_attack(model, x, labels, 0.02, num_iter=i )           
            prediction = smoothed_classifier.predict(x1, args.N, args.alpha, args.batch)
            after_time = time()
            cor += int(prediction == int(label))
            tot += int(prediction == prediction)
            time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        print("The final accuracy is", cor/tot)



