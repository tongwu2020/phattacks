import numpy as np
import torchvision
import cv2


def imshow(name,input1):
    
    input1 = torchvision.utils.make_grid(input1)
    inp = input1.cpu().numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)*255
    inp = inp.astype('uint8')
    
    cv2.imwrite("./image/"+name+".jpg",inp)
    
    