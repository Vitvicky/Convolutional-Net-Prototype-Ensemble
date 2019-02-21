import torch
from torch import tensor
from torch.utils.data import Dataset
from pandas import read_csv
from random import sample

DATASETS = {'fm', 'c10', 'svhn', 'cinic'}


class FashionMnist(Dataset):
    tensor_view = (1, 28, 28)
    train_test_split = 6000
    path = 'dataset/fashion-mnist_stream.csv'
    dataset = None

    def __init__(self, train=True):
        FashionMnist.dataset = read_csv(self.path, sep=',', header=None).values

        if train:
            dataset = FashionMnist.dataset[:self.train_test_split]
        else:
            dataset = FashionMnist.dataset[self.train_test_split:]

        self.data = []
        self.train = train
        self.label_set = set(dataset[:, -1].astype(int))

        for s in dataset:
            x = (tensor(s[:-1], dtype=torch.float) / 255).view(self.tensor_view)
            y = tensor(s[-1], dtype=torch.long)
            self.data.append((x, y))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class Cifar10(Dataset):
    tensor_view = (3, 32, 32)
    train_test_split = 6000
    path = 'dataset/cifar10_stream.csv'
    dataset = None

    def __init__(self, train=True):
        Cifar10.dataset = read_csv(self.path, sep=',', header=None).values

        if train:
            dataset = Cifar10.dataset[:self.train_test_split]
        else:
            dataset = Cifar10.dataset[self.train_test_split:]

        self.data = []
        self.train = train
        self.label_set = set(dataset[:, -1].astype(int))

        for s in dataset:
            x = (tensor(s[:-1], dtype=torch.float) / 255).view(self.tensor_view)
            y = tensor(s[-1], dtype=torch.long)
            self.data.append((x, y))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class SVHN(Dataset):
    tensor_view = (3, 32, 32)
    train_test_split = 6000
    path = 'dataset/SVHN_stream.csv'
    dataset = None

    def __init__(self, train=True):
        SVHN.dataset = read_csv(self.path, sep=',', header=None).values

        if train:
            dataset = SVHN.dataset[:self.train_test_split]
        else:
            dataset = SVHN.dataset[self.train_test_split:]

        self.data = []
        self.train = train
        self.label_set = set(dataset[:, -1].astype(int))

        for s in dataset:
            x = (tensor(s[:-1], dtype=torch.float) / 255).view(self.tensor_view)
            y = tensor(s[-1], dtype=torch.long)
            self.data.append((x, y))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class Cinic(Dataset):
    tensor_view = (3, 32, 32)
    train_test_split = 6000
    path = 'dataset/cinic_stream.csv'
    dataset = None

    def __init__(self, train=True):
        Cinic.dataset = read_csv(self.path, sep=',', header=None).values

        if train:
            dataset = Cinic.dataset[:self.train_test_split]
        else:
            dataset = Cinic.dataset[self.train_test_split:]

        self.data = []
        self.train = train
        self.label_set = set(dataset[:, -1].astype(int))

        for s in dataset:
            x = (tensor(s[:-1], dtype=torch.float) / 255).view(self.tensor_view)
            y = tensor(s[-1], dtype=torch.long)
            self.data.append((x, y))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class NoveltyDataset(Dataset):
    def __init__(self, dataset):
        self.data = list(dataset.data)
        self.label_set = set(dataset.label_set)

    def extend(self, buffer, percent):
        assert 0 < percent <= 1
        self.data.extend(sample(buffer, int(percent * len(buffer))))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
