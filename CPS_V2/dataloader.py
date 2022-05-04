from pandas.core.frame import DataFrame
import torch
import os
import pandas as pd
import torch
import h5py
import random
import numpy as np
from torch.utils.data import Dataset
import scipy.ndimage as sp

class LAHeart(Dataset):
    """ LA Dataset """
    def __init__(self, root=None, split='labelled_train', num=None, transform=None):
        self.root = os.path.join("./data")
        self.list = os.path.join("trainset.csv")
        self.split = split
        self.transform = transform
        self.sample_list = []

        if self.split == "labelled_train":
            csv = pd.read_csv(self.list)
            for i in range(num):
                filename = csv.iloc[i, 0]
                self.sample_list.append(filename)

        if self.split == "unlabelled_train":
            csv = pd.read_csv(self.list)
            for i in range(num):    
                filename = csv.iloc[i, 1]
                self.sample_list.append(filename)
                
    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        image_name = self.sample_list[idx]
        path = os.path.join("./data/{}/mri_norm2.h5".format(image_name))
        h5f = h5py.File(path, 'r')
        image = h5f['image'][:]
        label = h5f['label'][:]
        sample = {'image': image, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample


class RandomCrop(object):
    """
    Crop randomly the image in a sample
    Args:
    output_size (int): Desired output size
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # pad the sample if necessary
        if label.shape[0] <= self.output_size[0] or label.shape[1] <= self.output_size[1] or label.shape[2] <= \
                self.output_size[2]:
            pw = max((self.output_size[0] - label.shape[0]) // 2 + 3, 0)
            ph = max((self.output_size[1] - label.shape[1]) // 2 + 3, 0)
            pd = max((self.output_size[2] - label.shape[2]) // 2 + 3, 0)
            image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)
            label = np.pad(label, [(pw, pw), (ph, ph), (pd, pd)], mode='constant', constant_values=0)

        (w, h, d) = image.shape
        # if np.random.uniform() > 0.33:
        #     w1 = np.random.randint((w - self.output_size[0])//4, 3*(w - self.output_size[0])//4)
        #     h1 = np.random.randint((h - self.output_size[1])//4, 3*(h - self.output_size[1])//4)
        # else:
        w1 = np.random.randint(0, w - self.output_size[0])
        h1 = np.random.randint(0, h - self.output_size[1])
        d1 = np.random.randint(0, d - self.output_size[2])

        label = label[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        image = image[w1:w1 + self.output_size[0], h1:h1 + self.output_size[1], d1:d1 + self.output_size[2]]
        return {'image': image, 'label': label}


class RandomRotFlip(object):
    """
    Crop randomly flip the dataset in a sample
    Args:
    output_size (int): Desired output size
    """

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        k = np.random.randint(0, 4)
        image = np.rot90(image, k)
        label = np.rot90(label, k)
        axis = np.random.randint(0, 2)
        image = np.flip(image, axis=axis).copy()
        label = np.flip(label, axis=axis).copy()

        return {'image': image, 'label': label}


class RandomNoise(object):
    def __init__(self, mu=0, sigma=0.1):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        noise = np.clip(self.sigma * np.random.randn(image.shape[0], image.shape[1], image.shape[2]), -2*self.sigma, 2*self.sigma)
        noise = noise + self.mu
        image = image + noise
        return {'image': image, 'label': label}

class ToTensor(object):
    
    def __call__(self, sample):
        image = sample['image']
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        return {'image': torch.from_numpy(image), 'label': torch.from_numpy(sample['label']).long()}

class CreateOnehotLabel(object):
    def __init__(self, num_classes=2):
        self.num_classes = num_classes

    def __call__(self, predictions):
        label = predictions.detach().cpu().numpy()
        onehot_label = np.zeros((1, self.num_classes, label.shape[2], label.shape[3], label.shape[4]), dtype=np.float32)
        for i in range(self.num_classes):
            onehot_label[0, i, :, :, :] = (label[0][i] == i).astype(np.float32)
        return torch.from_numpy(onehot_label)
