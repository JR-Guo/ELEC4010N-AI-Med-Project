import os
import torch
import nrrd
import h5py
from os import walk



def read_h5(path):
    data = h5py.File(path, 'r')
    image = data['image'][:]
    label = data['label'][:]
    return image, label

class LAHeart(torch.utils.data.Dataset):
    def __init__(self, split='Training Set',label = True, transform=None):
        self.base_dir = f'./{split}/'
        self.label = label
        self.split = split
        self.path_list = os.listdir(self.base_dir)
        # print(self.path_list)
        self.transform = transform

    def __len__(self):
        if self.label and self.split == 'Training Set':
            return 32
        elif not self.label and self.split == 'Training Set':
            return (80-32)
        else:
            return 20

    def __getitem__(self, index):
        try:
            path = os.path.join(self.base_dir, self.path_list[index], \
                "mri_norm2.h5")
        except Exception:
            path = os.path.join(self.base_dir, self.path_list[1], \
                "mri_norm2.h5")
        image, label = read_h5(path)

        sample = {
            'image': image, 
            'label': label
        }
        if self.transform:
            sample = self.transform(sample)

        return sample
