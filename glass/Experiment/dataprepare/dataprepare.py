#prepare data

import xml.etree.ElementTree as ET
import cv2
import os
import numpy as np

import os.path



#dom = xml.dom.minidom.parse('/Users/wutong/Research/vgg-faces-utils/output/labels/a_j__buckley_00000003.xml')
#root = dom.documentElement
#bb = root.getElementsByTagName('xmin')[0]
#print("..................................")
#print(bb)
#print(bb.nodeValue)
#print(bb.nodeType)
#print(bb.ELEMENT_NODE)


def finddata(path,name):
	if os.path.isfile(path)==False:
		return 1

	tree = ET.parse(path)
	root = tree.getroot()
	file = root.find('filename').text

	if file==name:
		obj = root.find('object')
		bndbox = obj.find('bndbox')
	
		xmin = bndbox.find('xmin')
		ymin = bndbox.find('ymin')
		xmax = bndbox.find('xmax')
		ymax = bndbox.find('ymax')
	
		a = []
		a.append(xmin.text)
		a.append(ymin.text)
		a.append(xmax.text)
		a.append(ymax.text)

	#for child in root:
	    #print(child.tag, child.attrib)
		return a 



def cut(path,out):
	#print(path)
	img = cv2.imread(path)
	
	if img is None:
		return 1
	#cv2.imshow('image',img)
	xmin = int(out[0])
	xmax = int(out[2])
	ymin = int(out[1])
	ymax = int(out[3])
	#print(xmax-xmin, ymax-ymin)
	#print(img.shape)
	
	if (xmax-xmin)==img.shape[0] and (ymax-ymin)==img.shape[1]:
		return

	crop_img = img[ymin:ymax, xmin:xmax]
	cv2.imwrite(path[0:-4]+'.jpg',crop_img)
	print('success')




if __name__ == "__main__":
	path1 = '/Users/wutong/Research/vgg-faces-utils/output/images'
	path2 = '/Users/wutong/Research/vgg-faces-utils/output/labels'

	files= os.listdir(path1)
	s = []
	cout = 0 
	for file in files: 
		print(file)
		if file[-1]=='g':
			out = finddata('/Users/wutong/Research/vgg-faces-utils/output/labels/'+ file[0:-4]+'.xml',file)
			if out ==1:
				continue
			cut('/Users/wutong/Research/vgg-faces-utils/output/images/'+file,out)
			cout = cout+1


	print(cout)







