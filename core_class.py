from utils import extract_number
import re
import numpy as np
# Policy: branch model and leaf model don't store the model relationship itself


# TODO: BranchModel class
class BranchModel:
    def __init__(self, name, classes, clustering, cluster_num):
        self.name = name                    # string: used to for search trained .pkl file
        self.classes = classes              # arr of int: which classes are clustered by this branch
        self.clustering = clustering        # arr of int: results of clustering algo; len same as the above
        self.cluster_num = cluster_num      # classes are divided into n clusters
        self.need_training = True
        self.accuracy = -1

    def set_acc(self, acc):
        self.accuracy = acc

    def turn_train_state(self, bool_val):
        self.need_training = bool_val

    def add_classes(self, new_classes):
        # Here maybe have problems
        self.classes.extend(new_classes)
        self.need_training = True

    def update_cluster(self, new_cluster):
        if len(self.classes) == len(self.clustering):
            print('something wrong with update_cluster function')
        else:
            self.clustering = new_cluster

    def add_one_cluster(self, one_cluster):
        self.clustering = np.append(self.clustering, [one_cluster])
        self.need_training = True
        # self.clustering.append(one_cluster)

    def print_info(self):
        print()
        print('######################')
        print('name:{}'.format(self.name))
        print('accuracy:{}'.format(self.accuracy))
        print('classes:{}'.format(self.classes))
        print('clustering:{}'.format(self.clustering))
        print('######################')
        print()


# TODO: LeafModel class
class LeafModel:
    def __init__(self, name, classes, mapping):
        self.name = name
        self.classes = classes
        self.mapping = mapping
        self.need_training = False
        self.accuracy = -1

    def set_acc(self, acc):
        self.accuracy = acc

    def turn_train_state(self, bool_val):
        self.need_training = bool_val

    def add_classes(self, new_classes):

        start_idx = len(self.classes)

        self.classes.extend(new_classes)

        for true_class_idx in new_classes:
            self.mapping[str(start_idx)] = true_class_idx
            start_idx += 1
        self.need_training = True

    def print_info(self):
        print()
        print('####################')
        print('name:{}'.format(self.name))
        print('accuracy:{}'.format(self.accuracy))
        print('classes:{}'.format(self.classes))
        print('mapping:{}'.format(self.mapping))
        print('####################')
        print()


# TODO: LearningState class
class LearningState:
    def __init__(self):

        self.root_model = None         # string
        self.branch_model_list = []    # string
        self.leaf_model_list = []      # string

        self._old_classes = []         # int
        self._new_classes = []         # int

        self.leaf_class_map = dict()   # string -> int arr

        self.branch_leaf_map = dict()  # string -> string

        self.name_model_map = dict()   # string -> Model

    def set_root_model(self, root_model):
        self.root_model = root_model

    def _add_name(self, name, model):
        self.name_model_map[name] = model

    def _del_name(self, name):
        if name in self.name_model_map:
            self.name_model_map.pop(name)
        else:
            print('model name is not in the state list')

    def add_branch_model(self, branch_model):
        self.branch_model_list.append(branch_model.name)
        self.branch_leaf_map[branch_model.name] = []
        self._add_name(branch_model.name, branch_model)

    def add_leaf_model(self, leaf_model):
        if leaf_model.name in self.leaf_model_list:
            print('this leaf model has already in the state')

        self.leaf_model_list.append(leaf_model.name)
        self._add_name(leaf_model.name, leaf_model)
        self.leaf_class_map[leaf_model.name] = leaf_model.classes

    def init_state(self, leaf_model, classes):
        self.add_leaf_model(leaf_model)
        self._old_classes.extend(classes)
        self.leaf_class_map[leaf_model.name] = leaf_model.classes

    # input: model, model;
    # convert model to string and store in string
    def add_branch_leaf_map(self, branch_model, leaf_model):
        if branch_model.name in self.branch_model_list:
            self.branch_leaf_map[branch_model.name].append(leaf_model.name)
        else:
            self.branch_leaf_map[branch_model.name] = [leaf_model.name]

    def add_to_superclass(self, model_name, class_idx):  # model_name is the name of a leaf model
        # self.leaf_class_map[model_name].append(class_idx)
        if class_idx in self._new_classes:
            self._new_classes.remove(class_idx)
        else:
            print('something wrong with add to superclass function!!!!!')
        self._old_classes.append(class_idx)

        self.name_model_map[model_name].add_classes([class_idx])

        # TODO: add new class to the upper branch model [DONE]
        # TODO: Here only update the nearest branch model, not the rest
        # TODO: !!!!!! This code juebi tmd have problems !!!!
        # for branch_model in self.branch_model_list:
        #     if model_name in self.branch_leaf_map[branch_model]:
        #         self.name_model_map[branch_model].add_classes(class_idx)
        #
        #         for child_model_idx in range(len(self.branch_leaf_map[branch_model])):
        #             if self.branch_leaf_map[branch_model][child_model_idx] == model_name:
        #                 self.name_model_map[branch_model].add_one_cluster(child_model_idx)

        parent_idx = extract_number('p', model_name)
        layer = int(extract_number('l', model_name))
        cluster_id = int(extract_number('s', model_name))

        while layer > 0:

            # move to the upper layer
            layer = layer - 1

            # verify the constant combination
            if layer != len(parent_idx) - 2:
                print('Bug in core_class.py: add_to_superclass()')
                exit()

            for branch_name in self.branch_model_list:
                # Pattern for parent model
                s = re.match('l' + str(layer) + '-p' + parent_idx[:layer + 1] + '-s' + parent_idx[-1] + '.*',
                             branch_name)
                if s:
                    parent_model_name = s.group()
                    self.name_model_map[parent_model_name].add_classes([class_idx])
                    self.name_model_map[parent_model_name].add_one_cluster(cluster_id)
                    parent_idx = extract_number('p', parent_model_name)
                    cluster_id = extract_number('s', parent_model_name)
                    break

    def add_new_class(self, new_classes):
        self._new_classes.extend(new_classes)

    def check_new_classes_empty(self):
        return len(self._new_classes) == 0

    def transfer_new_classes(self):
        self._old_classes.extend(self._new_classes)
        self._new_classes.clear()

    # TODO: update parent branch model
    def update_flat(self, old_leaf_model, new_leaf_model):  # both Models

        self.leaf_class_map.pop(old_leaf_model.name)  # delete the old model in the system
        self.leaf_model_list.remove(old_leaf_model.name)
        # self.name_model_map.pop(old_leaf_model.name)

        self.add_leaf_model(new_leaf_model)
        self.leaf_class_map[new_leaf_model.name] = new_leaf_model.classes
        # self.name_model_map[new_leaf_model.name] = new_leaf_model

        for branch in self.branch_model_list:
            if old_leaf_model.name in self.branch_leaf_map[branch]:
                for child_model_idx in range(len(self.branch_leaf_map[branch])):
                    if self.branch_leaf_map[branch][child_model_idx] == old_leaf_model.name:
                        self.branch_leaf_map[branch][child_model_idx] = new_leaf_model.name

        if self.root_model == old_leaf_model.name:
            self.root_model = new_leaf_model.name

    # TODO: update parent branch model
    def update_clone(self, branch_model, leaf_list, old_leaf_model):
        # problem may occur because branch_model and old_leaf_model share the same name

        self.leaf_class_map.pop(old_leaf_model.name)  # delete the old model in the system
        self.leaf_model_list.remove(old_leaf_model.name)
        # self.name_model_map.pop(old_leaf_model.name)

        self.add_branch_model(branch_model)
        for leaf in leaf_list:
            self.add_branch_leaf_map(branch_model, leaf)
            self.add_leaf_model(leaf)

        # for branch in self.branch_model_list:
        #     if old_leaf_model.name in self.branch_leaf_map[branch]:
        #         for child_model_idx in range(len(self.branch_leaf_map[branch])):
        #             if self.branch_leaf_map[branch][child_model_idx] == old_leaf_model.name:
        #                 self.branch_leaf_map[branch] = [leaf.name for leaf in leaf_list]

        if self.root_model == old_leaf_model.name:
            self.root_model = branch_model.name

    def print_state(self):

        print()
        print('============================')
        print('root model:{}'.format(self.root_model))
        print('branch model list:{}'.format(self.branch_model_list))
        print('leaf model list:{}'.format(self.leaf_model_list))
        print()
        for key, value in self.branch_leaf_map.items():
            print('key:{} -> value:{}'.format(key, value))
        for key, value in self.leaf_class_map.items():
            print('key:{} -> value:{}'.format(key, value))

        print('total models:{}'.format([x for x in self.name_model_map]))

        for model in self.name_model_map.values():
            model.print_info()
        print('============================')
        print()

    def shutdown_all_training(self):
        for model in self.name_model_map.values():
            model.turn_train_state(False)

    def update_branch_name(self, old_name, new_name):
        if self.root_model == old_name:
            self.root_model = new_name
        if old_name == new_name:
            return

        self.name_model_map[old_name].name = new_name

        self.name_model_map[new_name] = self.name_model_map[old_name]
        self.name_model_map.pop(old_name)
        self.branch_leaf_map[new_name] = self.branch_leaf_map[old_name]
        self.branch_leaf_map.pop(old_name)

        self.branch_model_list.remove(old_name)
        self.branch_model_list.append(new_name)