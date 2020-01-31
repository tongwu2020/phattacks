
import os
import shutil
import re
path='/Users/wutong/Research/vgg-faces-utils/output/images'
path1='/Users/wutong/Research/vgg-faces-utils/class'

file_list = os.listdir(path1)

print(file_list)
id=[]

cout = 0 
for file in file_list:
	file1 = os.path.join(path1,'%s'%number)
	file1 = os.listdir(file_list[file])
	print(file1) 


