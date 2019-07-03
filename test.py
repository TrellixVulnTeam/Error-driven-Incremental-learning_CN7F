import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch import tensor

import copy
import time
import os

from os.path import join

import configs
from Models.models import initialize_model
from Dataset.increm_animals import ClassSplit, FlexAnimalSet, BranchSet
from Dataset.load import load_datasets
from core_class import BranchModel, LeafModel, LearningState
from train_module import train_model, validate, MySampler

from sklearn.cluster import spectral_clustering
import numpy as np
from utils import extract_number

import argparse

class_split = ClassSplit(45, [15, 5, 5, 5, 5, 5, 5], random_seed=0)

testset = FlexAnimalSet(join('Dataset', 'CIFAR100-animal'), False, class_split, [x for x in range(15)], None)

# testset = BranchSet(join('Dataset', 'CIFAR100-animal'), False, class_split, [[0,1,2,4,5,6,7,8,9,10,12,13,14],
#                                                                              [3, 11]], None)

test_loader = DataLoader(testset, 128, shuffle=False)

model = torch.load(join(configs.trained, 'l0-p0-s0.pkl'))

criterion = nn.CrossEntropyLoss()
validate(test_loader, model, criterion)

