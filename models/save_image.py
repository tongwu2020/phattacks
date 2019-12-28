import numpy as np
import torchvision
import cv2

def save_image(name,input1):
    
    input1 = torchvision.utils.make_grid(input1)
    inp = input1.cpu().numpy().transpose((1, 2, 0))
    mean = np.array([129.1863, 104.7624,93.5940  ])
    inp = inp + mean
    inp = np.clip(inp, 0, 255)
    inp = inp.astype('uint8')
    
    cv2.imwrite("./image_test/"+name+".jpg",inp)
    
    
def save_image2(name,input1):
    
    #input1 = torchvision.utils.make_grid(input1)
    inp = input1.transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)*255
    inp = inp.astype('uint8')
    
    cv2.imwrite("./image_test/"+name+".jpg",inp)
    
    