import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils import data
from torchvision import datasets, transforms

from pypc import utils


class MNIST(datasets.MNIST):
    def __init__(self, train, size=None, scale=None, normalize=False):
        transform = _get_transform(normalize=normalize, mean=(0.1307), std=(0.3081))
        super().__init__("./data/mnist", download=True, transform=transform, train=train)
        self.scale = scale
        if size is not None:
            self._reduce(size)

    def __getitem__(self, index):
        data, target = super().__getitem__(index)
        data = _to_vector(data)
        target = _one_hot(target)
        if self.scale is not None:
            target = _scale(target, self.scale)
        return data, target

    def _reduce(self, size):
        self.data = self.data[0:size]
        self.targets = self.targets[0:size]


class SVHN(datasets.SVHN):
    def __init__(self, train, size=None, scale=None, normalize=False):
        if normalize:
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            )
        else:
            transform = transforms.Compose([transforms.ToTensor()])
        super().__init__("./data/svhn", download=True, transform=transform, train=train)
        self.scale = scale
        if size is not None:
            self._reduce(size)

    def __getitem__(self, index):
        data, target = super().__getitem__(index)
        data = _to_vector(data)
        target = _one_hot(target)
        if self.scale is not None:
            target = _scale(target, self.scale)
        return data, target

    def _reduce(self, size):
        self.data = self.data[0:size]
        self.targets = self.targets[0:size]


class CIFAR10(datasets.CIFAR10):
    def __init__(self, train, size=None, scale=None, normalize=False):
        if normalize:
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            )
        else:
            transform = transforms.Compose([transforms.ToTensor()])
        super().__init__("./data/cifar10", download=True, transform=transform, train=train)
        self.scale = scale
        if size is not None:
            self._reduce(size)

    def __getitem__(self, index):
        data, target = super().__getitem__(index)
        data = _to_vector(data)
        target = _one_hot(target)
        if self.scale is not None:
            target = _scale(target, self.scale)
        return data, target

    def _reduce(self, size):
        self.data = self.data[0:size]
        self.targets = self.targets[0:size]


class CIFAR100(datasets.CIFAR100):
    def __init__(self, train, size=None, scale=None, normalize=False):
        if normalize:
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            )
        else:
            transform = transforms.Compose([transforms.ToTensor()])
        super().__init__("./data/cifar100", download=True, transform=transform, train=train)
        self.scale = scale
        if size is not None:
            self._reduce(size)

    def __getitem__(self, index):
        data, target = super().__getitem__(index)
        data = _to_vector(data)
        target = _one_hot(target, n_classes=100)
        if self.scale is not None:
            target = _scale(target, self.scale)
        return data, target

    def _reduce(self, size):
        self.data = self.data[0:size]
        self.targets = self.targets[0:size]


class FashionMNIST(datasets.FashionMNIST):
    def __init__(self, train, size=None, normalize=False):
        transform = _get_transform(normalize=normalize, mean=(0.5), std=(0.5))
        super().__init__("./data/mnist", download=True, transform=transform, train=train)
        if size is not None:
            self._reduce(size)

    def __getitem__(self, index):
        data, target = super().__getitem__(index)
        data = _to_vector(data)
        target = _one_hot(target)
        return data, target

    def _reduce(self, size):
        self.data = self.data[0:size]
        self.targets = self.targets[0:size]


def get_dataloader(dataset, batch_size):
    dataloader = data.DataLoader(dataset, batch_size, shuffle=True, drop_last=True)
    return list(map(_preprocess_batch, dataloader))


def accuracy(pred_labels, true_labels):
    batch_size = pred_labels.size(0)
    correct = 0
    for b in range(batch_size):
        if torch.argmax(pred_labels[b, :]) == torch.argmax(true_labels[b, :]):
            correct += 1
    return correct / batch_size


def plot_imgs(img_preds, path):
    imgs = img_preds.cpu().detach().numpy()
    imgs = imgs[0:10, :]
    imgs = [np.reshape(imgs[i, :], [28, 28]) for i in range(imgs.shape[0])]
    _, axes = plt.subplots(2, 5)
    axes = axes.flatten()
    for i, img in enumerate(imgs):
        axes[i].imshow(img, cmap="gray")
    plt.savefig(path)
    plt.close("all")


def _preprocess_batch(batch):
    batch[0] = utils.set_tensor(batch[0])
    batch[1] = utils.set_tensor(batch[1])
    return (batch[0], batch[1])


def _get_transform(normalize=True, mean=(0.5), std=(0.5)):
    transform = [transforms.ToTensor()]
    if normalize:
        transform + [transforms.Normalize(mean=mean, std=std)]
    return transforms.Compose(transform)


def _one_hot(labels, n_classes=10):
    arr = torch.eye(n_classes)
    return arr[labels]


def _scale(targets, factor):
    return targets * factor + 0.5 * (1 - factor) * torch.ones_like(targets)


def _to_vector(batch):
    batch_size = batch.size(0)
    return batch.reshape(batch_size, -1).squeeze()
