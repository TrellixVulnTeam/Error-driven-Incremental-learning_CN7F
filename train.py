
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

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--randn', metavar='N', type=int, default=1,
                    help='random seed for the whole system')
args = parser.parse_args()

torch.manual_seed(args.randn)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def extend_leaf_model(state, leaf_model, class_split, num_superc):  # string, list
    print('start training leaf model')
    if not leaf_model.need_training:
        print('no new class coming in, quit...')
        return
    old_leaf_name = leaf_model.name
    current_layer, old_model_id, old_parent = \
        extract_number('l', old_leaf_name), extract_number('s', old_leaf_name), extract_number('p', old_leaf_name)

    class_array = leaf_model.classes

    old_leaf_model = copy.deepcopy(leaf_model)

    # net, _ = initialize_model('resnet34', num_classes=len(class_array), feature_extract=False)
    net = torch.load(join(configs.trained, old_leaf_name+'.pkl'))

    trainset = FlexAnimalSet(join('Dataset', 'CIFAR100-animal'), True, class_split, class_array, None)
    testset = FlexAnimalSet(join('Dataset', 'CIFAR100-animal'), False, class_split, class_array, None)

    num_classes = len(trainset.class_enum)
    num_features = net.fc.in_features

    net.fc = nn.Linear(num_features, num_classes)

    dataset = dict()
    dataset['train'] = trainset
    dataset['val'] = testset

    flat_model_name = 'l{}-p{}-s{}'.format(current_layer, old_parent, old_model_id)
    # if os.path.exists(join(configs.trained, flat_model_name + '.pkl')):
    #     flat_net = torch.load(join(configs.trained, flat_model_name + '.pkl'))
    # else:
    flat_net, best_acc = train(parse_args(), flat_model_name, dataset, model=net)
    flat_net = copy.deepcopy(flat_net)

    cluster_labels, flat_acc = clustering(flat_model_name, testset, num_superc)
    print('cluster_labels:{}'.format(cluster_labels))

    leaf_net_ls = []
    superclass = []  # 2d-array
    for_test = []
    leaf_name_ls = []

    if num_classes > 10:
        for i in range(num_superc):

            superclass.append([class_array[idx] for idx in range(len(cluster_labels)) if cluster_labels[idx] == i])
            for_test.append([idx for idx in range(len(cluster_labels)) if cluster_labels[idx] == i])
            print('superclass_{}:{}'.format(i, superclass[-1]))

            num_subclass = len(superclass[-1])

            model_name = 'l{}-p{}{}-s{}'.format(current_layer+1, old_parent, old_model_id, i)
            leaf_name_ls.append(model_name)
            train_set = FlexAnimalSet(join('Dataset', 'CIFAR100-animal'), True, class_split, superclass[-1])
            test_set = FlexAnimalSet(join('Dataset', 'CIFAR100-animal'), False, class_split, superclass[-1])

            sub_net = net
            sub_net.fc = nn.Linear(num_features, num_subclass)
            best_acc = 'Not given'
            # if os.path.exists(join(configs.trained, model_name+'.pkl')):
            #     net = torch.load(join(configs.trained, model_name+'.pkl'))
            # else:
            net, best_acc = train(parse_args(), model_name,
                                  make_dataset_dict(train_set, test_set), model=sub_net)
            print('branch_{}_accuracy:{}'.format(i, best_acc))

            leaf_net_ls.append(copy.deepcopy(net))

        test_loader = DataLoader(testset, 1, shuffle=True)

        # below is comparing two models' accuracy (flat & clone)
        running_corrects = 0.0
        branch_corrects = 0.0
        for inputs, labels in test_loader:
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = flat_net(inputs)
            # print('output:{}'.format(outputs))
            s = torch.nn.Softmax(dim=0)
            softmax = s(outputs[0])

            best_score = 0
            best_score_idx = 0
            for i in range(num_superc):

                temp = [softmax[idx].item() for idx in range(len(softmax)) if cluster_labels[idx] == i]
                if len(temp) < 2:
                    print('temp:{}'.format(temp))
                    print('cluster:{}'.format(cluster_labels))
                    print('outputs:{}'.format(outputs))
                    print('softmax:{}'.format(softmax))
                    print('classify number:{}'.format(flat_net.fc.out_features))
                    return
                average_score = np.average(temp)

                if average_score > best_score:
                    best_score = average_score
                    best_score_idx = i

            sub_outputs = leaf_net_ls[best_score_idx](inputs)
            _, sub_preds = torch.max(sub_outputs, 1)
            # superclass[best_score_idx] = torch.LongTensor(superclass[best_score_idx])
            preds = for_test[best_score_idx][sub_preds]

            preds = torch.tensor(preds).cuda()
            running_corrects += torch.sum(preds == labels.data)

            for s in for_test:
                if labels.data in torch.LongTensor(s).cuda() and preds in torch.LongTensor(s).cuda():
                    branch_corrects += 1

        clone_acc = running_corrects.item() / len(testset)
        branch_acc = branch_corrects / len(testset)
        print()
        print('clone accuracy: {}'.format(clone_acc))
        print('branch accuracy: {}'.format(branch_acc))
        print('flat accuracy: {}'.format(flat_acc))
        print()
    else:
        clone_acc = 0

    if num_classes < 7 or flat_acc-0.02 > clone_acc:  # policy: if num_class is less than 3, use flat increment only
        new_leaf_model = LeafModel(flat_model_name, class_array, leaf_model.mapping)
        state.update_flat(old_leaf_model, new_leaf_model)

        if state.root_model == old_leaf_name:
            state.set_root_model(new_leaf_model.name)
    else:
        leaf_model_ls = []
        for i in range(num_superc):

            temp_map = dict()

            for j in range(len(superclass[i])):
                temp_map[str(j)] = superclass[i][j]
            leaf_model = LeafModel(leaf_name_ls[i], superclass[i], temp_map)
            leaf_model_ls.append(copy.deepcopy(leaf_model))

        branch_model = BranchModel(flat_model_name, class_array, cluster_labels, num_superc)
        state.update_clone(branch_model, leaf_model_ls, old_leaf_model)

        return


def make_dataset_dict(trainset, testset):
    output = dict()
    output['train'] = trainset
    output['val'] = testset
    return output


def clustering(model_path, testset, num_superc):

    test_loader = DataLoader(testset, 1, shuffle=True)

    model = torch.load(join(configs.trained, model_path+'.pkl'))
    # num_classes = test_set.num_classes
    # print('num of classes: {}'.format(num_classes))
    # num_ftrs = model.fc.in_features
    # print('last layer parameters')
    #
    # prev_param_dict = dict()
    # for namel, param in model.fc.named_parameters():
    #     if param.requires_grad:
    #         prev_param_dict[namel] = param
    #
    # model.fc = nn.Linear(num_ftrs, num_classes)
    #
    # param_weight = torch.randn(new_classes, num_ftrs)/100
    # param_weight = nn.Parameter(param_weight)
    # param_weight = param_weight.cuda()
    # param_weight = torch.cat((prev_param_dict['weight'], param_weight))
    # model.fc.weight = nn.Parameter(param_weight)
    #
    # param_bias = torch.randn(new_classes)/100
    # param_bias = nn.Parameter(param_bias)
    # param_bias = param_bias.cuda()
    # param_bias = torch.cat((prev_param_dict['bias'], param_bias))
    # model.fc.bias = nn.Parameter(param_bias)

    model.eval()
    running_corrects = 0.0
    num_classes = len(testset.class_enum)
    cm = np.zeros((num_classes, num_classes))

    for inputs, labels in test_loader:
        inputs = inputs.cuda()
        labels = labels.cuda()
        outputs = model(inputs)

        _, preds = torch.max(outputs, 1)
        running_corrects += torch.sum(preds == labels.data)
        cm[labels.data][preds] += 1

    accuracy = running_corrects*100/len(testset)
    # print('accuracy: {}'.format(running_corrects*100/len(testset)))
    clustering = spectral_clustering(cm,
                                     n_clusters=num_superc,
                                     assign_labels="kmeans",
                                     random_state=0)
    # print(clustering)

    # superclass_split = clustering.labels_

    # branch_model = torch.load(configs.trained + '/' + flat_increm_model)
    #
    # left_leaf_model = torch.load(configs.trained + '/' + flat_increm_model)
    # left_leaf_model.fc = nn.Linear(num_ftrs, sum([x for x in superclass_split if x == 0]))
    #
    # right_leaf_model = torch.load(configs.trained + '/' + flat_increm_model)
    # right_leaf_model.fc = nn.Linear(num_ftrs, sum([x for x in superclass_split if x == 1]))
    return clustering, int(accuracy)/100


def train(args: object, saved_name: object, dataset: object, model: object = None) -> object:

    if not os.path.exists(configs.trained):
        os.makedirs(configs.trained, exist_ok=True)

    if not os.path.exists('./log'):
        os.makedirs('./log', exist_ok=True)

    batch_size = args.batch_size
    n_epochs = args.n_epochs
    learning_rate = args.learning_rate
    momentum = args.momentum
    model_name = args.model_name
    feature_extract = False
    name = saved_name
    use_pretrained = args.pretrain

    train_set = dataset['train']
    val_set = dataset['val']

    try:
        classes = len(train_set.class_enum)
    except:
        classes = 100

    num_classes = classes

    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    test_loader = DataLoader(val_set, batch_size, shuffle=False)

    if model:
        net = model
    else:
        net, _ = initialize_model(model_name, num_classes, feature_extract, use_pretrained=use_pretrained)

    net = net.cuda()

    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

    criterion = nn.CrossEntropyLoss()

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    best_accuracy = 0.0
    best_model = None

    for i, epoch in enumerate(range(n_epochs)):
        # train_loader = DataLoader(train_set, batch_size, shuffle=False, sampler=MySampler(train_set, i))
        train_model(train_loader, net, criterion, optimizer, epoch)
        accuracy = validate(test_loader, net, criterion)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = copy.deepcopy(net)

    # trained_model, best_acc = train_model(name, net, loader_dict, optimizer, scheduler, criterion, n_epochs,
    #                                       batch_size, is_inception=(model_name == "inception"))
    torch.save(best_model, join(configs.trained, name+'.pkl'))

    return net, best_accuracy


# def train_model(name,
#                 net,
#                 loader_dict,
#                 optimizer,
#                 scheduler,
#                 criterion,
#                 n_epochs,
#                 batch_size,
#                 test=True,
#                 is_inception=False):
#
#     since = time.time()
#
#     val_acc_history = []
#
#     best_model_wts = copy.deepcopy(net.state_dict())
#
#     best_acc = 0.0
#
#     cudnn.benchmark = True
#
#     log = ''
#
#     for epoch in range(n_epochs):
#         print('Epoch {}/{}, Learning rate: {}'.format(epoch+1, n_epochs, optimizer.param_groups[0]['lr']))
#         log += 'Epoch {}/{}'.format(epoch+1, n_epochs) + '\n'
#
#         print('-' * 15)
#         log += '-' * 15 + '\n'
#
#         # each epoch has a traing and validation phase
#         phases = ['train']
#         if test:
#             phases.append('val')
#
#         for phase in phases:
#             if phase == 'train':
#                 net.train()
#             else:
#                 net.eval()
#
#             running_loss = 0.0
#             running_corrects = 0
#
#             idx_num = 1
#             for idx, dd in enumerate(loader_dict[phase]):
#                 # print('this is batch {}'.format(idx))
#                 inputs, labels = dd
#                 inputs = inputs.cuda()
#                 labels = labels.to(torch.device("cuda"), dtype=torch.int64)
#
#                 optimizer.zero_grad()
#
#                 with torch.set_grad_enabled(phase == 'train'):
#
#                     if is_inception and phase == 'train':
#
#                         outputs, aux_outputs = net(inputs)
#                         loss1 = criterion(outputs, labels)
#                         loss2 = criterion(aux_outputs, labels)
#                         loss = loss1 + 0.4 * loss2
#                     else:
#                         outputs = net(inputs)
#                         loss = criterion(outputs, labels)
#
#                     _, preds = torch.max(outputs, 1)
#
#                     # backward + optimize only if in training phase
#                     if phase == 'train':
#                         loss.backward()
#                         optimizer.step()
#
#                 running_loss += loss.item() * inputs.size(0)
#                 running_corrects += torch.sum(preds == labels.data)
#                 # statistics
#                 if (idx + 1) % 100 == 0:
#                     print('Batch:{}, Loss:{}, Accuracy:{}%'.format(idx + 1, running_loss / (batch_size * idx_num * 100),
#                                                                    running_corrects / (batch_size * idx_num)))
#                     idx_num += 1
#
#             epoch_loss = running_loss / len(loader_dict[phase].dataset)
#             epoch_acc = running_corrects.double() / len(loader_dict[phase].dataset)
#
#             print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
#             log += '{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc) + '\n'
#
#             if phase == 'val' and epoch_acc > best_acc:
#                 best_acc = epoch_acc
#                 best_model_wts = copy.deepcopy(net.state_dict())
#             if phase == 'val':
#                 val_acc_history.append(epoch_acc)
#
#         scheduler.step()
#         print()
#
#     time_elapsed = time.time() - since
#
#     print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
#     log += 'Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60) + '\n'
#
#     print('Best val Acc: {:4f}'.format(best_acc))
#     log += 'Best val Acc: {:4f}'.format(best_acc) + '\n'
#
#     # load best model weights
#     net.load_state_dict(best_model_wts)
#     torch.save(net, configs.trained + '/' + name + '.pkl')
#
#     f = open('log/' + name + '.txt', 'w')
#     f.write(log)
#     f.close()
#
#     return net, best_acc


def model_set_forward(learning_state, input):

    root_model_name = learning_state.root_model
    current_model = learning_state.name_model_map[root_model_name]  # an instance of branch model or leaf model

    while True:
        model = torch.load(join(configs.trained, current_model.name + '.pkl'))
        output = model(input)
        # print('picking model {} with class number {}'.format(current_model.name, current_model.classes))
        s = nn.Softmax(dim=0)
        softmax = s(output[0])
        # if current_model.model_type == 'branch':
        if isinstance(current_model, BranchModel):
            # print()
            # print('current model name:{}'.format(current_model.name))
            # print('output softmax:{}'.format(softmax))
            # print('model clustering:{}'.format(current_model.clustering))
            # print()

            category = 0
            max_value = max(softmax)
            for idx, value in enumerate(softmax):
                if value == max_value:
                    category = idx
            # results = []
            # for i in range(current_model.cluster_num):
            #     temp = []
            #     for j in range(len(current_model.clustering)):
            #         if current_model.clustering[j] == i:
            #             temp.append(softmax[j].item())
            #     results.append(np.average(temp))  # a list of length = num_classes: start from 7-th element
            #
            # max_num = max(results)
            # category = 0
            # for i in range(len(results)):
            #     if results[i] == max_num:
            #         category = i
            current_model_name = learning_state.branch_leaf_map[current_model.name][category]
            current_model = learning_state.name_model_map[current_model_name]
        else:  # in leaf model
            _, pred = torch.max(output, 1)
            original_pred = current_model.mapping[str(pred.item())]
            return original_pred


def new_class_split(state, class_split, new_class_array):

    if state.check_new_classes_empty():
        return

    testset = FlexAnimalSet(join('Dataset', 'CIFAR100-animal'), False, class_split, new_class_array)
    test_loader = DataLoader(testset, 1, shuffle=True)

    cm = np.zeros((len(new_class_array), new_class_array[0]))  # new class * old class

    for input, label in test_loader:
        input = input.cuda()
        label = label.cuda()

        # TODO: change api of model_set_forward func: [DONE]
        output = model_set_forward(state, input)
        cm[label][output] += 1

    return cm


# def incremental_learning(root_model, class_split, new_class_array, leaf_class_map):
def incremental_learning(state, class_split, new_class_array):
    state.add_new_class(new_class_array)

    # TODO: change api of new class split function [DONE]
    confusion_matrix = new_class_split(state, class_split, new_class_array)

    class_idx = 0
    origin_map = copy.deepcopy(state.leaf_class_map)
    for line in confusion_matrix:

        leaf_hit = dict()
        for key, value in origin_map.items():
            leaf_hit[key] = sum([line[idx] for idx in value])

        max_hit = 0
        max_model_name = ''

        for key, value in leaf_hit.items():
            if value > max_hit:
                max_hit = value
                max_model_name = key

        # max_model_name is the name of a leaf model
        state.add_to_superclass(max_model_name, new_class_array[class_idx])

        class_idx += 1

    state.transfer_new_classes()

    # TODO: change the api of extend_leaf_model[DONE]

    temp_ls = copy.deepcopy(state.leaf_model_list)
    print('leaf model list:{}'.format(temp_ls))
    for model_name in temp_ls:
        # print('model_name:{}'.format(model_name))
        leaf_model = state.name_model_map[model_name]
        extend_leaf_model(state, leaf_model, class_split, 2)

    origin_branch_ls = copy.deepcopy(state.branch_model_list)

    print('ready to train branch!!')
    for branch_name in origin_branch_ls:
        print('train branch!!')
        train_branch(state, branch_name, class_split)

    state.shutdown_all_training()

    return state


def train_branch(state, branch_model_name, class_split):
    print('start re-training the branch')
    branch_model = state.name_model_map[branch_model_name]
    if not branch_model.need_training:
        print('no need training this branch')
        return
    net = torch.load(join(configs.trained, branch_model_name+'.pkl'))
    old_num_class = net.fc.out_features

    if len(branch_model.classes) == old_num_class:
        print('something wrong in train_branch method')

    classes = copy.deepcopy(branch_model.classes)
    clustering = copy.deepcopy(branch_model.clustering)
    cluster_num = copy.deepcopy(branch_model.cluster_num)

    superc = []
    for cl_idx in range(cluster_num):
        sub = []
        for idx in range(len(clustering)):
            if clustering[idx] == cl_idx:
                sub.append(classes[idx])
        superc.append(sub)

    # trainset = FlexAnimalSet(join('Dataset', 'CIFAR100-animal'), True, class_split, classes, None)
    # testset = FlexAnimalSet(join('Dataset', 'CIFAR100-animal'), False, class_split, classes, None)

    trainset = BranchSet(join('Dataset', 'CIFAR100-animal'), True, class_split, superc, None)
    testset = BranchSet(join('Dataset', 'CIFAR100-animal'), False, class_split, superc, None)

    new_model_name = 'l{}-p{}-s{}'.format(extract_number('l', branch_model_name),
                                          extract_number('p', branch_model_name),
                                          extract_number('s', branch_model_name))

    num_ftr = net.fc.in_features
    # net.fc = nn.Linear(num_ftr, len(branch_model.classes))
    net.fc = nn.Linear(num_ftr, cluster_num)
    train(parse_args(), new_model_name,
          make_dataset_dict(trainset, testset), model=net)

    state.update_branch_name(branch_model_name, new_model_name)

    # state.name_model_map[new_model_name].need_training = False


def parse_args():
    class Args:
        def __init__(self, batch_size, n_epochs, learning_rate, momentum, model_name, pretrain):
            self.batch_size = batch_size
            self.n_epochs = n_epochs
            self.learning_rate = learning_rate
            self.momentum = momentum
            self.model_name = model_name
            self.pretrain = pretrain
    args = Args(batch_size=100,
                n_epochs=5,
                learning_rate=0.001,
                momentum=0.9,
                model_name='resnet34',
                pretrain=True)
    return args


def main(randn_seed):
    start_time = time.time()
    if randn_seed == 100:
        class_split = ClassSplit(45, [], random_sort=[2,  14,  25, 42, 31,
                                                      0,  16,  30, 32, 40,
                                                      3,   4,   5,  7, 10,
                                                      1,  20,  21, 39, 43,
                                                      6,   9,   8, 15, 19,
                                                      12, 13,  22, 36, 41,
                                                      11, 23,  35, 44, 37,
                                                      18, 28,  24, 38, 33,
                                                      17, 26,  27, 29, 34])
    elif randn_seed == 101:
        class_split = ClassSplit(45, [], random_sort=[0,  16, 30, 32, 40,
                                                      2,  14, 25, 42, 31,
                                                      3,   4,  5,  7, 10,
                                                      1,  20, 21, 39, 43,
                                                      6,   9,  8, 15, 19,
                                                      12, 13, 22, 36, 41,
                                                      11, 23, 35, 44, 37,
                                                      17, 26, 27, 29, 34,
                                                      18, 28, 24, 38, 33])
    elif randn_seed == 50:
        class_split = ClassSplit(45, [], random_sort=[0,  2,  3,   1,   6,  12, 11, 17, 18,
                                                      16, 14, 4,  20,   9,  13, 23, 26, 28,
                                                      30, 25, 5,  21,   8,  22, 35, 27, 24,
                                                      32, 42, 7,  39,  15,  36, 44, 29, 38,
                                                      40, 31, 10, 43,  19,  41, 37, 34, 33])
    else:
        class_split = ClassSplit(45, [15, 5, 5, 5, 5, 5, 5], random_seed=randn_seed)

    init_name = 'l0-p0-s0'

    trainset = FlexAnimalSet(join('Dataset', 'CIFAR100-animal'), True, class_split, [x for x in range(5)], None)
    testset = FlexAnimalSet(join('Dataset', 'CIFAR100-animal'), False, class_split, [x for x in range(5)], None)

    # trainset, testset, _ = load_datasets('CIFAR100-animal')

    train(parse_args(), init_name, make_dataset_dict(trainset, testset))

    init_map = dict()

    for i in range(5):
        init_map[str(i)] = i

    root_model = LeafModel(init_name, [x for x in range(5)], init_map)

    state = LearningState()
    state.set_root_model(root_model.name)
    state.init_state(root_model, [x for x in range(5)])

    for i in range(1, 9):
        print('coming batch: {}'.format(i))
        state = incremental_learning(state, class_split, [x for x in range(i*5, i*5 + 5)])
        state.print_state()

    # print('coming batch: 1')
    # state = incremental_learning(state, classSplit, [5, 6, 7, 8, 9])
    # state.print_state()
    # print('coming batch: 2')
    # state = incremental_learning(state, classSplit, [10, 11, 12, 13, 14])
    # state.print_state()
    # print('coming batch: 3')
    # state = incremental_learning(state, classSplit, [15, 16, 17, 18, 19])
    # state.print_state()
    # print('coming batch: 4')
    # state = incremental_learning(state, classSplit, [20, 21, 22, 23, 24])
    # state.print_state()
    # print('coming batch: 5')
    # state = incremental_learning(state, classSplit, [25, 26, 27, 28, 29])
    # state.print_state()
    # print('coming batch: 6')
    # state = incremental_learning(state, classSplit, [30, 31, 32, 33, 34])
    # state.print_state()
    # print('coming batch: 7')
    # state = incremental_learning(state, classSplit, [35, 36, 37, 38, 39])
    # state.print_state()
    # print('coming batch: 8')
    # state = incremental_learning(state, classSplit, [40, 41, 42, 43, 44])
    # state.print_state()

    print('finish training')
    end_time = time.time()
    print('total training time:{}s'.format(end_time-start_time))

    all_testset = FlexAnimalSet(join('Dataset', 'CIFAR100-animal'), False, class_split, [x for x in range(45)], None)
    all_testloader = DataLoader(all_testset, 1, shuffle=False)

    corrects = 0.0
    for X, y in all_testloader:
        X, y = X.cuda(), y.cuda()
        pred = model_set_forward(state, X)
        if pred == y:
            corrects += 1

    accuracy = corrects / len(all_testset)
    print('accuracy:{}'.format(accuracy))

    if not os.path.exists('results'):
        os.makedirs('results')
    with open('results/randn-{}.txt'.format(randn_seed), 'a') as f:
        f.write(str(accuracy) + '\n')
        f.write('similarity:{}'.format(class_split.similar) + '\n')
        f.write('random seed:{}'.format(args.randn))
        f.write('\n')
        f.write('============================\n')
        f.write('root model:{}\n'.format(state.root_model))
        f.write('branch model list:{}\n'.format(state.branch_model_list))
        f.write('leaf model list:{}\n'.format(state.leaf_model_list))
        f.write('\n')
        for key, value in state.branch_leaf_map.items():
            f.write('key:{} -> value:{}\n'.format(key, value))
        for key, value in state.leaf_class_map.items():
            f.write('key:{} -> value:{}\n'.format(key, value))

        f.write('total models:{}\n'.format([x for x in state.name_model_map]))

        for model in state.name_model_map.values():
            if isinstance(model, BranchModel):
                f.write('\n')
                f.write('######################\n')
                f.write('name:{}\n'.format(model.name))
                f.write('classes:{}\n'.format(model.classes))
                f.write('clustering:{}\n'.format(model.clustering))
                f.write('######################\n')
                f.write('\n')
            else:
                f.write('\n')
                f.write('####################\n')
                f.write('name:{}\n'.format(model.name))
                f.write('classes:{}\n'.format(model.classes))
                f.write('mapping:{}\n'.format(model.mapping))
                f.write('####################\n')
                f.write('\n')
        f.write('============================\n')


if __name__ == "__main__":
    main(0)
    # for i in range(8):
    #     main(i)
    #     main(i)
    #     main(i)

    # main(50)
    # main(100)
    # main(100)
    # main(101)
    # main(101)
    # main(101)

