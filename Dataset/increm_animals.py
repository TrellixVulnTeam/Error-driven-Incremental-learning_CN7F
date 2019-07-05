from __future__ import print_function

from PIL import Image
from os.path import join
import os

import torch.utils.data as data
from torchvision.datasets import ImageFolder
from torchvision import transforms
import random

import copy
import numpy as np
from scipy.io import loadmat


class ClassSplit:

    def __init__(self, total_class, split, random_seed=None, random_sort=None):
        self.total_class = total_class
        self.split = split
        if random_seed:
            self.random_seed = random_seed
        else:
            self.random_seed = 813

        if random_sort:
            self.random_sort = random_sort
        else:
            random.seed(self.random_seed)
            class_enum = [i for i in range(total_class)]
            random.shuffle(class_enum)
            self.random_sort = class_enum

        self.similar = self.get_sim_idx()
        print(self.similar)

    def get_sim_idx(self):
        superclass = [[2,  14,  25, 31, 42],
                      [0,  16,  30, 32, 40],
                      [3,   4,   5,  7, 10],
                      [1,  20,  21, 39, 43],
                      [6,   8,   9, 15, 19],
                      [12, 13,  22, 36, 41],
                      [11, 23,  35, 37, 44],
                      [18, 24,  28, 33, 38],
                      [17, 26,  27, 29, 34]]

        def to_matrix(l, n):
            return [l[i:i + n] for i in range(0, len(l), n)]

        split = to_matrix(self.random_sort, 5)

        for ls in split:
            ls.sort()

        def compare_sim(arr1, arr2):
            result = 0
            for elem1 in arr1:
                for elem2 in arr2:
                    if elem1 == elem2:
                        result += 1
            if result == 1:
                return 0
            else:
                return result

        print(superclass)
        print(split)
        total_sim = 0
        for sub1 in split:
            for sub2 in superclass:
                total_sim += compare_sim(sub1, sub2)
        return total_sim


class BranchSet(data.Dataset):
    def __init__(self,
                 root,
                 train,
                 class_split,     # an instance of ClassSplit
                 superclass_set,  # array of array
                 transform=None,
                 crop=False):

        self.root = root
        self.train = train
        self.classSplit = class_split
        self.superclass = superclass_set
        self.class_enum = [x for x in range(len(self.superclass))]

        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0,
                                     np.array([63.0, 62.1, 66.7]) / 255.0),
            ])
            if train:
                self.transform = transforms.Compose([
                    transforms.Pad(4, padding_mode='reflect'),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32),
                    self.transform
                ])

        if train:
            self.phase = 'train'
        else:
            self.phase = 'test'

        temp_total = ImageFolder(join(self.root, self.phase))

        class_to_idx = temp_total.class_to_idx

        idx_to_class = dict()

        for key, value in class_to_idx.items():
            idx_to_class[str(value)] = key

        label_ls = []

        start_idx = 0

        self.return_tuple = []

        for sub in self.superclass:
            for idx in sub:
                original_idx = self.classSplit.random_sort[idx]
                class_name = idx_to_class[str(original_idx)]
                label_ls.append(class_name)

                self.return_tuple.extend([(join(class_name, fname),
                                           start_idx)
                                          for fname in os.listdir(join(self.root,
                                                                       self.phase,
                                                                       class_name))])
            start_idx += 1

        print(label_ls)

        self.crop = crop
        if crop:
            self.bbox = loadmat(join(self.root, self.phase, 'annotations'))

    def __len__(self):
        return len(self.return_tuple)

    def __getitem__(self, item):
        image_name, target_class = self.return_tuple[item]
        image_path = join(self.root, self.phase, image_name)
        image = Image.open(image_path).convert('RGB')

        if self.crop:
            image = image.crop(self.bbox[image_name.lower()][0])

        if self.transform:
            image = self.transform(image)

        return image, target_class


class FlexAnimalSet(data.Dataset):

    def __init__(self,
                 root,
                 train,
                 class_split,   # an instance of ClassSplit
                 required_arr,  # An array of new class index
                 transform=None,
                 crop=False):

        self.root = root
        self.train = train
        self.classSplit = class_split
        self.required_arr = required_arr
        self.class_enum = self.required_arr

        if transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                # transforms.Resize(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0,
                                     np.array([63.0, 62.1, 66.7]) / 255.0),
            ])

        if train:
            self.phase = 'train'
        else:
            self.phase = 'test'

        temp_total = ImageFolder(join(self.root, self.phase))

        class_to_idx = temp_total.class_to_idx

        idx_to_class = dict()

        for key, value in class_to_idx.items():
            idx_to_class[str(value)] = key

        label_ls = []

        start_idx = 0

        self.return_tuple = []
        self.idx_map = dict()

        for idx in self.required_arr:  # here idx is the primary idx
            original_idx = self.classSplit.random_sort[idx]
            class_name = idx_to_class[str(original_idx)]
            label_ls.append(class_name)
            self.idx_map[str(start_idx)] = idx

            self.return_tuple.extend([(join(class_name, fname),
                                       start_idx)
                                      for fname in os.listdir(join(self.root,
                                                                   self.phase,
                                                                   class_name))])
            start_idx += 1

        print(label_ls)

        self.crop = crop
        if crop:
            self.bbox = loadmat(join(self.root, 'annotations'))

    def __len__(self):
        return len(self.return_tuple)

    def __getitem__(self, item):
        image_name, target_class = self.return_tuple[item]
        image_path = join(self.root, self.phase, image_name)
        image = Image.open(image_path).convert('RGB')

        if self.crop:
            image = image.crop(self.bbox[image_name.lower()][0])
        if self.transform:
            image = self.transform(image)

        return image, target_class


