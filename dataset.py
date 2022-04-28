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
        self.base_dir = f'./Dataset/{split}/'
        self.label = label
        self.split = split
        self.path_list = os.listdir(self.base_dir)
        # print(self.path_list)
        self.transform = transform

    def __len__(self):
        if self.label and self.split == 'Training Set':
            return 16
        elif not self.label and self.split == 'Training Set':
            return (80-16)
        else:
            return 20

    def __getitem__(self, index):
        path = os.path.join(self.base_dir, self.path_list[index], \
            "mri_norm2.h5")
        image, label = read_h5(path)

        sample = {
            'image': image, 
            'label': label
        }
        if self.transform:
            sample = self.transform(sample)

        return sample


# def main():
#     temp_dataset = LAHeart(split = 'Training Set')
#     for i in range(16):
#         print(temp_dataset[i]['image'].shape, temp_dataset[i]['label'].shape)
#     return

# if '__main__':
#     main()