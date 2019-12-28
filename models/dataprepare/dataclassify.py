import os
import shutil
import re
path='/Users/wutong/Research/vgg-faces-utils/output/images'
path1='/Users/wutong/Research/vgg-faces-utils/class'

file_list = os.listdir(path)

id=[]
for i in range(len(file_list)):
    id.append(file_list[i].split('_0000')[0])
id=set(id)

print(id)

sort_folder_number = list(id)
for number in sort_folder_number:
    new_folder_path = os.path.join(path1,'%s'%number)
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)

print(len(file_list))
for i in range(len(file_list)):
    old_file_path = os.path.join(path,file_list[i])
    fid=file_list[i].split('_0000')[0]
    new_file_path = os.path.join(path1,'%s'%(fid),file_list[i])
    shutil.move(old_file_path,new_file_path)




