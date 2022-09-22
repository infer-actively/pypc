import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils import data
from torchvision import datasets, transforms

from pypc import utils


class MNIST(datasets.MNIST):
    def __init__(self, train, size=None, scale=None, normalize=False):
        """
        Load MNIST dataset, scale to range [0,1], optionally normalise with mean = 0 and std dev = 1,
        optionally set label scale factor, optionally limit size of dataset

        :param train: True for training data, False for test data
        :param size: Number of samples to keep in the dataset
        :param scale: Scale factor for one-hot labels (e.g. scale=1 => 0 or 1, scale=0.5 => 0.25 or 0.75,
        scale=0.1 => 0.45 or 0.55)
        :param normalize: True to normalize
        """
        transform = _get_transform(normalize=normalize, mean=(0.1307), std=(0.3081))  # Transform to mean=0, std=1
        super().__init__("./data/mnist", download=True, transform=transform, train=train)
        self.scale = scale
        if size is not None:
            self._reduce(size)

    def __getitem__(self, index):
        """
        Return image (data) and label (target) with image converted from (1,28,28) to (784,) and label converted to
        one-hot encoding (optionally scaled)

        :param index: Index
        :return: image, label
        """
        data, target = super().__getitem__(index)
        data = _to_vector(data)
        target = _one_hot(target)
        if self.scale is not None:
            target = _scale(target, self.scale)
        return data, target

    def _reduce(self, size):
        """
        Crop the dataset

        :param size: Maximum sample number to retain
        """
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
    """
    Create PyTorch DataLoader for given dataset and batch size, perform pre-processing to move data onto the selected
    cpu/cuda device with dtype=torch.float32, and return a list containing samples and labels

    :param dataset: PyTorch Dataset
    :param batch_size: Batch size
    :return: List of tuples with index 0 containing samples and index 1 containing labels
    """
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
    """
    Pre-process a batch to move data onto the selected cpu/cuda device with dtype=torch.float32

    :param batch: List of Tensor objects with index 0 containing samples and index 1 containing labels
    :return: Pre-processed batch as tuple of Tensor objects
    """
    batch[0] = utils.set_tensor(batch[0])
    batch[1] = utils.set_tensor(batch[1])
    return (batch[0], batch[1])


def _get_transform(normalize=True, mean=0.0, std=1.0):
    """
    Define transformation to convert PIL image or numpy.ndarray to tensor with optional normalization

    :param normalize: True or False
    :param mean: Input mean (after scaling to range [0,1])
    :param std: Input std dev (after scaling to range [0,1])
    :return: Transformation
    """
    transform = [transforms.ToTensor()]
    if normalize:
        transform += [transforms.Normalize(mean=mean, std=std)]
    return transforms.Compose(transform)


def _one_hot(labels, n_classes=10):
    """
    Convert categorical label to one-hot encoding (trick is to index an identity matrix) NOTE: Only used for individual
    labels so consider changing parameter name to 'label')

    :param labels: Categorical label
    :param n_classes: Number of classes (categories)
    :return: One-hot encoded label
    """
    arr = torch.eye(n_classes)
    return arr[labels]


def _scale(targets, factor):
    """
    Scale one-hot labels (targets) according to:
    scaled = 0.5 + factor x (original - 0.5)
    (e.g. scale=1 => 0 or 1, scale=0.5 => 0.25 or 0.75, scale=0.1 => 0.45 or 0.55)

    :param targets: Labels
    :param factor: Scale factor
    :return: Scaled labels
    """
    return targets * factor + 0.5 * (1 - factor) * torch.ones_like(targets)


def _to_vector(batch):
    """
    Convert batch of 2D images to vector format NOTE: Currently only used for single images so naming is confusing
    :param batch: Image or batch of images
    :return: Image or batch of images in vector format
    """
    batch_size = batch.size(0)
    return batch.reshape(batch_size, -1).squeeze()
