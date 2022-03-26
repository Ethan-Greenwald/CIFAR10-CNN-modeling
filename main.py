import torch
from torchvision.datasets import CIFAR100, CelebA
from torchvision.transforms import ToTensor

# train_data = CIFAR100(root='data', train=True, transform=ToTensor, download=True)
# test_data = CIFAR100(root='data', train=False, transform=ToTensor, download=True)

train_data = CelebA(root='celeb_data', split='train', target_type='attr', transform=ToTensor, download=True)