#this file is used to save some images 

import numpy as np
import torchvision
import cv2

def save_image(name,input1):
    
    input1 = torchvision.utils.make_grid(input1)
    inp = input1.cpu().numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    inp = inp*255
    inp = inp.astype('uint8')
    
    cv2.imwrite("./images/"+name+".jpg",cv2.cvtColor(inp, cv2.COLOR_RGB2BGR))
    


def save_image2(name,input1):
    
    #input1 = torchvision.utils.make_grid(input1)
    inp = input1.transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)*255
    inp = inp.astype('uint8')
    
    cv2.imwrite("./images/"+name+".jpg",inp)
    
    