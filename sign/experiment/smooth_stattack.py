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
from save_image import save_image 
import os
import copy
import cv2
from physical_attack import untarget_attack



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Predict on many examples')
    parser.add_argument("model", type=str, help="test_model")
    parser.add_argument("sigma", type=float, help="noise hyperparameter")
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

    #for i in [ 10,100,1000 ]:
    for i in [ 1000 ]:
        cor = 0
        tot = 0 
        for k in dataloaders['test']:
            (x, label) = k
            x = x.to(device)
            labels = label.to(device)
    
            before_time = time()
            x1 = untarget_attack(model, x, labels, 0.02, num_iter=i )
            
            prediction = smoothed_classifier.predict(x1, args.N, args.alpha, args.batch)
            #print("label is ", label, "prediction is ", prediction)
            after_time = time()
            cor += int(prediction == int(label))
            tot += int(prediction == prediction)
            print(prediction, int(label))
            if int(prediction != int(label)):
                save_image('11uatt'+str(tot)+str(prediction),x.data)
                save_image('11att'+str(tot)+str(prediction),x1.data)
            time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
            # log the prediction and whether it was correct
            #print("{}\t{}\t{}\t{}\t{}".format(i, label, prediction, cor, time_elapsed))
            print(cor/tot)



