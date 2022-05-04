import os
import torch
import numpy as np
import PIL
from PIL import Image
import nrrd
import h5py
from os import walk

def read_h5(path):
    data = h5py.File(path, 'r')
    image = data['image'][:]
    label = data['label'][:]
    return image, label

base_dir = "./Dataset/Training Set"
path_list = os.listdir(base_dir)
path = os.path.join(base_dir, path_list[1], \
    "mri_norm2.h5")
image, label = read_h5(path)

for i in range(9):
    temp = image[:,:,i*8]
    templa = label[:,:,i*8]
    
    for index in range(temp.shape[0]):
            for j in range(temp.shape[1]):
                if temp[index][j] != 0:
                    temp[index][j] = 32
                if templa[index][j] != 0:
                    templa[index][j] = 32
    # break
    im = Image.fromarray(np.int8(temp)).convert('L')
    impath = "./image/" + str(i) + ".png"
    im.save(impath)
        
    pr = Image.fromarray(np.int8(templa)).convert('L')
    prpath = "./pred/" + str(i) + ".png"
    pr.save(prpath)

print("DONE")