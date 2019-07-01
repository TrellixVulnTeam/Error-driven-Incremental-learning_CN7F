import os

from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder
from Dataset.stanford_dogs_data import dogs

import configs

import numpy as np


def load_datasets(set_name, input_size=224):
    if set_name == 'mnist':
        train_dataset = datasets.MNIST(root=os.path.join(configs.imagesets, 'MNIST'),
                                       train=True,
                                       transform=transforms.ToTensor(),
                                       download=True)
        test_dataset = datasets.MNIST(root=os.path.join(configs.imagesets, 'MNIST'),
                                      train=False,
                                      transform=transforms.ToTensor())

        classes = train_dataset.classes

    elif set_name == 'CIFAR100-animal':
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        train_dataset = ImageFolder('Dataset/CIFAR100-animal/train', transform=transform)
        test_dataset = ImageFolder('Dataset/CIFAR100-animal/test', transform=transform)

        classes = train_dataset.classes

    elif set_name == 'CIFAR10-png':
        transform = transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        train_dataset = ImageFolder('Dataset/CIFAR10-png/train', transform=transform)
        test_dataset = ImageFolder('Dataset/CIFAR10-png/test', transform=transform)

        classes = train_dataset.classes

    elif set_name == 'CIFAR100-png':
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        train_dataset = ImageFolder('Dataset/CIFAR100-png/train', transform=transform)
        test_dataset = ImageFolder('Dataset/CIFAR100-png/test', transform=transform)

        classes = train_dataset.classes

    elif set_name == 'stanford_dogs':
        input_transforms = transforms.Compose([
            transforms.RandomResizedCrop(input_size, ratio=(1, 1.3)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])

        train_dataset = dogs(root=configs.imagesets,
                             train=True,
                             cropped=True,
                             transform=input_transforms,
                             download=True)
        test_dataset = dogs(root=configs.imagesets,
                            train=False,
                            cropped=True,
                            transform=input_transforms,
                            download=True)

        classes = train_dataset.classes

        print("Training set stats:")
        train_dataset.stats()
        print("Testing set stats:")
        test_dataset.stats()

    elif set_name == 'CIFAR10':
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(root='./Dataset', train=True,
                                    download=True, transform=transform)

        test_dataset = datasets.CIFAR10(root='./Dataset', train=False,
                                   download=True, transform=transform)

        classes = ('plane', 'car', 'bird',
                   'cat', 'deer', 'dog',
                   'frog', 'horse', 'ship', 'truck')

    elif set_name == 'CIFAR100':

        # transform = transforms.Compose([
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0,
                                 np.array([63.0, 62.1, 66.7]) / 255.0),
        ])

        transform_train = transforms.Compose([
                transforms.Pad(4, padding_mode='reflect'),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32),
                transform_test
            ])

        train_dataset = datasets.CIFAR100(root='./Dataset', train=True,
                                          download=True, transform=transform_train)

        test_dataset = datasets.CIFAR100(root='./Dataset', train=False,
                                         download=True, transform=transform_test)
        classes = None
    else:
        return None, None

    return train_dataset, test_dataset, classes
