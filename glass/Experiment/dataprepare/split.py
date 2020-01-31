import os
import shutil
import re

import random





def moveFile(fileDir,tarDir):
        pathDir = os.listdir(fileDir)    #取图片的原始路径
        filenumber=len(pathDir)
        rate=1  #自定义抽取图片的比例，比方说100张抽10张，那就是0.1
        picknumber=int(filenumber*rate) #按照rate比例从文件夹中取一定数量图片
        sample = random.sample(pathDir, picknumber)  #随机选取picknumber数量的样本图片
        print (sample)
        print("success")
        for name in sample:
                shutil.move(fileDir+"/"+name, tarDir)
        return

if __name__ == '__main__':
	path='/Users/wutong/Research/vgg-faces-utils/class'
	file_list = os.listdir(path)

	for i in range(len(file_list)):
		if file_list[i][0]=="a":
			class1 = os.path.join(path,file_list[i])
			class2 = os.path.join('/Users/wutong/Research/vgg-faces-utils/class/test',file_list[i])

			#print(file_list[i],class1)
			#b = os.path.exists(class2)
			#print(b)
			#os.mkdir(class2)
			moveFile(class1,class2)

 
# 
#if __name__ == '__main__':
#
#	path='/Users/wutong/Research/vgg-faces-utils/class'
#
#	file_list = os.listdir(path)
#
#	for i in range(len(file_list)):
#		class1 = os.path.join(path,file_list[i])
#		png_list = os.listdir(class1)
#		print(len(png_list))
#		#for j in range(len(png_list)):
#			#print(png_list[j])
#	

