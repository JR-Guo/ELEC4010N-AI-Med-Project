import os
import torch
import PIL
from PIL import Image
import nrrd
import numpy as np
import h5py
from os import walk

def read_h5(path):
    data = h5py.File(path, 'r')
    image = data['image'][:]
    label = data['label'][:]
    return image, label

base_dir = "./Training Set"
path_list = os.listdir(base_dir)
path = os.path.join(base_dir, path_list[5], \
    "mri_norm2.h5")
image, label = read_h5(path)

for i in range(9):
  
    temp = image[:,:,i*8]
    templa = label[:,:,i*8]

    for index in range(temp.shape[0]):
        for j in range(temp.shape[1]):
            # Can use for testing visualisations
            # temp[index][j] *= 25.5
            # templa[index][j] *= 25.5
            if temp[index][j] != 0:
                temp[index][j] = 255
            if templa[index][j] != 0:
                templa[index][j] = 255
                
    im = Image.fromarray(np.int8(temp)).convert('L')
    impath = "./image/" + str(i) + ".png"
    im.save(impath)
    
    pr = Image.fromarray(np.int8(templa)).convert('L')
    prpath = "./pred/" + str(i) + ".png"
    pr.save(prpath)

# print(np.count_nonzero(image[:,:,74]))
# print(np.count_nonzero(label))
# print(np.count_nonzero(label[:,:,74]))
# temp = image[:,:,74]
# templa = label[:,:,74]
# print(temp.shape)
# print(np.max(temp))
# print(templa.shape)
# print(np.max(templa))
# im = Image.fromarray(temp).convert('L')
# impath = "./image/" + str(32) + ".png"
# im.save(impath)
# pr = Image.fromarray(np.byte(templa)).convert('L')
# prpath = "./pred/" + str(32) + ".png"
# pr.save(prpath)

