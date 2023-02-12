import math

from torchvision import datasets
import torch
import collections
import random




class ImagesOnlyCelebA(datasets.CelebA):
    """
    A CelebA dataset containing only the image
    """

    def __getitem__(self, index: int):
        image, target = super().__getitem__(index)
        return image


class BalancedAttributedCelebASet(datasets.CelebA):
    def __init__(self, *args, **kwargs):
        self.selected_attr = kwargs.pop('selected_attr')
        self.group_size = kwargs.pop("group_size")
        super().__init__(*args, **kwargs)
        self.selected_attr_indices = []
        for attr in self.selected_attr:
            self.selected_attr_indices.append(self.attr_names.index(attr))
        self.selected_attr_indices = torch.tensor(self.selected_attr_indices)

        attr_groups = collections.defaultdict(list)
        for i,attr in enumerate(self.attr):
            attr_values = torch.index_select(attr, 0, self.selected_attr_indices)
            attr_groups[tuple(attr_values.tolist())].append(i)

        self.smallest_group_size = super().__len__()
        for k, v in attr_groups.items():
            self.smallest_group_size = min(self.smallest_group_size,len(v))
            assert len(v) >= self.group_size, "group size too big"
        self.attr_groups = attr_groups

    def __len__(self):
        return math.floor(self.smallest_group_size/self.group_size)

    def __getitem__(self, item):
        items_to_get = []
        for k,v in self.attr_groups.items():
            items_to_get += v[item*self.group_size:(item+1)*(self.group_size)]
        items = []
        for i in items_to_get:
            pic, _ = super(BalancedAttributedCelebASet, self).__getitem__(i)
            items.append(pic.unsqueeze(0))
        return torch.cat(items)

class SpecificAttributedCelebASet(datasets.CelebA):
    def __init__(self, *args, **kwargs):
        self.selected_attr = kwargs.pop('selected_attr')
        self.group_size = kwargs.pop("group_size")
        super().__init__(*args, **kwargs)
        self.selected_attr_indices = []
        for attr in self.selected_attr:
            self.selected_attr_indices.append(self.attr_names.index(attr))
        self.selected_attr_indices = torch.tensor(self.selected_attr_indices)
        self.good_indices=[]
        for i,attr in enumerate(self.attr):
            attr_values = torch.index_select(attr, 0, self.selected_attr_indices)
            if torch.all(attr_values==1).item():
                self.good_indices.append(i)

    def __len__(self):
        return math.floor(len(self.good_indices)/self.group_size)

    def __getitem__(self, item):
        items = []
        for i in self.good_indices[item*self.group_size:(item+1)*(self.group_size)]:
            pic, _ = super(SpecificAttributedCelebASet, self).__getitem__(i)
            items.append(pic.unsqueeze(0))
        return torch.cat(items)

class SpecificAttributedNotCelebASet(datasets.CelebA):
    def __init__(self, *args, **kwargs):
        self.selected_attr = kwargs.pop('selected_attr')
        self.group_size = kwargs.pop("group_size")
        super().__init__(*args, **kwargs)
        self.selected_attr_indices = []
        for attr in self.selected_attr:
            self.selected_attr_indices.append(self.attr_names.index(attr))
        self.selected_attr_indices = torch.tensor(self.selected_attr_indices)
        self.good_indices=[]
        for i,attr in enumerate(self.attr):
            attr_values = torch.index_select(attr, 0, self.selected_attr_indices)
            if torch.all(attr_values!=1).item():
                self.good_indices.append(i)

    def __len__(self):
        return math.floor(len(self.good_indices)/self.group_size)

    def __getitem__(self, item):
        items = []
        for i in self.good_indices[item*self.group_size:(item+1)*(self.group_size)]:
            pic, _ = super(SpecificAttributedNotCelebASet, self).__getitem__(i)
            items.append(pic.unsqueeze(0))
        return torch.cat(items)

class GroupImageOnlyCelebA(datasets.CelebA):
    """
    A CelebA dataset containing only the image
    """

    def __init__(self, *args, **kwargs):
        self.group_size = kwargs.pop("group_size")
        super().__init__(*args, **kwargs)

    def __len__(self):
        return math.floor(super(GroupImageOnlyCelebA, self).__len__()/self.group_size)

    def __getitem__(self, index: int):
        items = []
        for i in range(self.group_size):
            pic, _ = super(GroupImageOnlyCelebA, self).__getitem__(index*self.group_size+i)
            items.append(pic.unsqueeze(0))
        return torch.cat(items)

class BinaryLabelCelebA(datasets.CelebA):
    def __getitem__(self, item):
        image, attr = super().__getitem__(item)
        attr[attr==-1] = 0
        return image, attr
    def __len__(self):
        return min(10000,super().__len__())