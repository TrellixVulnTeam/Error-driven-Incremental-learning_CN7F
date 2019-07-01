# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.backends.cudnn as cudnn
# from torch.utils.data import DataLoader
# from torch import tensor
#
# import copy
# import time
# import argparse
# import os
#
# from os.path import join
#
# import configs
# from Models.models import initialize_model
# from Dataset.increm_animals import ClassSplit, FlexAnimalSet
# from Dataset.load import load_datasets
# from core_class import BranchModel, LeafModel, LearningState
# from train_module import train_model, validate
#
# from sklearn.cluster import spectral_clustering
# import numpy as np
#
#
# def extend_leaf_model(state, leaf_model, class_split, num_superc):  # string, list
#     old_leaf_name = leaf_model.name
#
#     class_array = leaf_model.classes
#
#     old_leaf_model = copy.deepcopy(leaf_model)
#
#     model = initialize_model('resnet34', num_classes=len(class_array))
#
#     trainset = FlexAnimalSet(join('Dataset', 'CIFAR100-animal'), True, class_split, class_array, None)
#     testset = FlexAnimalSet(join('Dataset', 'CIFAR100-animal'), False, class_split, class_array, None)
#
#     num_classes = len(trainset.class_enum)
#     num_features = model.fc.in_features
#
#     dataset = dict()
#     dataset['train'] = trainset
#     dataset['val'] = testset
#
#     flat_model_name = 'flat-{}'.format(num_classes)
#     if os.path.exists(join(configs.trained, flat_model_name + '.pkl')):
#         flat_net = torch.load(join(configs.trained, flat_model_name + '.pkl'))
#     else:
#         flat_net, best_acc = train(parse_args(), flat_model_name, dataset, model=model)
#
#     cluster_labels, flat_acc = clustering(flat_model_name, testset, num_superc)
#     print('cluster_labels:{}'.format(cluster_labels))
#
#     leaf_net_ls = []
#     superclass = []
#     leaf_name_ls = []
#     for i in range(num_superc):
#         superclass[i] = [idx for idx in range(len(cluster_labels)) if cluster_labels[idx] == i]
#         print('superclass_{}:{}'.format(i, superclass))
#
#         num_subclass = len(superclass)
#         # net = initialize_model('resnet34', num_classes=num_subclass)
#
#         model_name = 'clone-{}-{}'.format(i, num_subclass)
#         leaf_name_ls.append(model_name)
#         train_set = FlexAnimalSet(join('Dataset', 'CIFAR100-animal'), True, class_split, superclass)
#         test_set = FlexAnimalSet(join('Dataset', 'CIFAR100-animal'), False, class_split, superclass)
#         # if os.path.exists(join(configs.trained, model_name + '.pkl')):
#         #     net = torch.load(join(configs.trained, model_name + '.pkl'))
#         # else:
#         net, best_acc = train(parse_args(), model_name,
#                                                  make_dataset_dict(train_set, test_set))
#         print('branch_{}_accuracy:{}'.format(i, best_acc))
#
#         leaf_net_ls.append(copy.deepcopy(net))
#
#         test_loader = DataLoader(testset, 1, shuffle=True)
#
#     running_corrects = 0.0
#     for inputs, labels in test_loader:
#         inputs, labels = inputs.cuda(), labels.cuda()
#         outputs = flat_net(inputs)
#
#         best_score = 0
#         best_score_idx = 0
#         for i in range(num_superc):
#             temp = [outputs[0][idx].item() for idx in range(len(outputs[0])) if cluster_labels[idx] == i]
#             temp.sort()
#             temp = temp[5:]
#             average_score = np.average(temp)
#
#             if average_score > best_score:
#                 best_score = average_score
#                 best_score_idx = i
#
#         sub_outputs = leaf_net_ls[best_score_idx](inputs)
#         _, sub_preds = torch.max(sub_outputs, 1)
#         superclass_0 = torch.LongTensor(superclass_0)
#         preds = superclass[sub_preds]
#
#         preds = preds.cuda()
#         running_corrects += torch.sum(preds == labels.data)
#
#     clone_acc = running_corrects.item() / len(testset)
#     print('clone accuracy: {}'.format(clone_acc))
#     print('flat accuracy: {}'.format(flat_acc))
#
#     if flat_acc > clone_acc:
#         new_leaf_model = LeafModel(flat_model_name, class_array, leaf_model.mapping)
#         state.update_flat(old_leaf_model, new_leaf_model)
#     else:
#         leaf_model_ls = []
#         for i in range(num_superc):
#
#             temp_map = dict()
#
#             for j in range(len(superclass[i])):
#                 temp_map[str(i)] = superclass[i][j]
#             leaf_model = LeafModel(leaf_name_ls[i], superclass[i], temp_map)
#             leaf_model_ls.append(leaf_model)
#
#
#         branch_model = BranchModel(flat_model_name, class_array, cluster_labels, 2)
#         state.update_clone(branch_model, leaf_model_ls, old_leaf_model)
#
#         os.remove(join(configs.trained, flat_model_name + '.pkl'))
#         return
