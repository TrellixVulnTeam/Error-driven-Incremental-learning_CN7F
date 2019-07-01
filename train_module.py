import argparse
import os
import time
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Sampler

import torchvision
import torchvision.transforms as transforms

from Models import *
from Models import models
from Dataset.increm_animals import ClassSplit, FlexAnimalSet
from Dataset.load import load_datasets

from os.path import join
import configs


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')
    parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=100, type=int, metavar='N',
                        help='mini-batch size (default: 128),only used for train')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float, metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W',
                        help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=30, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('-ct', '--cifar-type', default='100', type=int, metavar='CT',
                        help='10 for cifar10,100 for cifar100 (default: 10)')

    return parser


best_prec = 0


def train_general():
    global args, best_prec
    parser = parse_args()
    args = parser.parse_args()
    use_gpu = torch.cuda.is_available()

    # Model building
    print('=> Building model...')
    if use_gpu:
        # model can be set to anyone that I have defined in models folder
        # note the model should match to the cifar type !

        # model = resnet20_cifar()
        # model = resnet32_cifar()
        # model = resnet44_cifar()
        # model = resnet110_cifar()
        # model = preact_resnet110_cifar()
        # model = resnet164_cifar(num_classes=45)
        # model = resnet1001_cifar(num_classes=100)
        # model = preact_resnet164_cifar(num_classes=100)
        # model = preact_resnet1001_cifar(num_classes=100)

        # model = wide_resnet_cifar(depth=26, width=10, num_classes=100)

        # model = resneXt_cifar(depth=29, cardinality=16, baseWidth=64, num_classes=100)

        # model = densenet_BC_cifar(depth=190, k=40, num_classes=100)

        model, _ = models.initialize_model('resnet101', 45, False, use_pretrained=True)

        # mkdir a new folder to store the checkpoint and best model
        if not os.path.exists('result'):
            os.makedirs('result')
        fdir = 'result/resnet20_cifar10'
        if not os.path.exists(fdir):
            os.makedirs(fdir)

        # adjust the lr according to the model type
        # if isinstance(model, (ResNet_Cifar, PreAct_ResNet_Cifar)):
        if True:
            model_type = 1
        # elif isinstance(model, Wide_ResNet_Cifar):
        #     model_type = 2
        # elif isinstance(model, (ResNeXt_Cifar, DenseNet_Cifar)):
        #     model_type = 3
        else:
            print('model type unrecognized...')
            return

        model = nn.DataParallel(model).cuda()
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        cudnn.benchmark = True
    else:
        print('Cuda is not available!')
        return

    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint "{}"'.format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec = checkpoint['best_prec']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Data loading and preprocessing
    # CIFAR10
    if args.cifar_type == 10:
        print('=> loading cifar10 data...')
        normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])

        train_dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=2)

        test_dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]))
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)
    # CIFAR100
    else:
        print('=> loading cifar100 data...')
        normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])

        train_dataset = torchvision.datasets.CIFAR100(
            root='./data',
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=2)

        test_dataset = torchvision.datasets.CIFAR100(
            root='./data',
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]))
        testloader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)

    # trainset, testset, _ = load_datasets('CIFAR100-animal')
    classSplit = ClassSplit(45, [15, 5, 5, 5, 5, 5, 5], random_seed=17)
    trainset = FlexAnimalSet(join('Dataset', 'CIFAR100-animal'), True, classSplit, [x for x in range(30)], None)
    testset = FlexAnimalSet(join('Dataset', 'CIFAR100-animal'), False, classSplit, [x for x in range(30)], None)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2,
                                              sampler=MySampler(trainset))
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, model_type)

        # train for one epoch
        train_model(trainloader, model, criterion, optimizer, epoch)

        # evaluate on test set
        prec = validate(testloader, model, criterion)

        # remember best precision and save checkpoint
        is_best = prec > best_prec
        best_prec = max(prec, best_prec)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec': best_prec,
            'optimizer': optimizer.state_dict(),
        }, is_best, fdir)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_model(trainloader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()

    end = time.time()
    for i, (input, target) in enumerate(trainloader):
        # measure data loading time
        data_time.update(time.time() - end)

        input, target = input.cuda(), target.cuda()

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec = accuracy(output, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # if i % args.print_freq == 0:
        # if i % 100 == 0:
        #     print('Epoch: [{0}][{1}/{2}]\t'
        #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #           'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
        #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #           'Prec {top1.val:.3f}% ({top1.avg:.3f}%)'.format(
        #            epoch, i, len(trainloader), batch_time=batch_time,
        #             data_time=data_time, loss=losses, top1=top1))


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input, target = input.cuda(), target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec = accuracy(output, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # if i % args.print_freq == 0:
            # if i % 100 == 0:
            #     print('Test: [{0}/{1}]\t'
            #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            #           'Prec {top1.val:.3f}% ({top1.avg:.3f}%)'.format(
            #            i+1, len(val_loader), batch_time=batch_time, loss=losses,
            #            top1=top1))

    print(' * Prec {top1.avg:.3f}% '.format(top1=top1))

    return top1.avg


def save_checkpoint(state, name):
    filepath = os.path.join(configs.trained, name+'.pkl')
    torch.save(state, filepath)


def adjust_learning_rate(optimizer, epoch, model_type):
    """For resnet, the lr starts from 0.1, and is divided by 10 at 80 and 120 epochs"""
    if model_type == 1:
        if epoch < 30:
            lr = args.lr
        elif epoch < 60:
            lr = args.lr * 0.1
        else:
            lr = args.lr * 0.01
        # if epoch < 80:
        #     lr = args.lr
        # elif epoch < 120:
        #     lr = args.lr * 0.1
        # else:
        #     lr = args.lr * 0.01
    elif model_type == 2:
        if epoch < 60:
            lr = args.lr
        elif epoch < 120:
            lr = args.lr * 0.2
        elif epoch < 160:
            lr = args.lr * 0.04
        else:
            lr = args.lr * 0.008
    elif model_type == 3:
        if epoch < 150:
            lr = args.lr
        elif epoch < 225:
            lr = args.lr * 0.1
        else:
            lr = args.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class MySampler(Sampler):
    def __init__(self, data_source, random_seed, replacement=False, num_samples=None):
        super().__init__(data_source)
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.random_seed = random_seed

        if not isinstance(self.replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(self.replacement))

        if self._num_samples is not None and not replacement:
            raise ValueError("With replacement=False, num_samples should not be specified, "
                             "since a random permute will be performed.")

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(self.num_samples))

    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        n = len(self.data_source)
        import random
        random.seed(self.random_seed)
        ls = [x for x in range(n)]
        random.shuffle(ls)
        ls_t = torch.LongTensor(ls)
        if self.replacement:
            return iter(torch.randint(high=n, size=(self.num_samples,), dtype=torch.int64).tolist())
        # return iter(torch.randperm(n).tolist())
        return iter(ls_t.tolist())

    def __len__(self):
        return self.num_samples


if __name__ == '__main__':
    train_general()
