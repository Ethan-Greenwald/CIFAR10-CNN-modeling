import training
import testing
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets
from torchvision.transforms import ToTensor, ToPILImage
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader


#GOAL: Implement two networks: one that can detect facial attributes and one dedicated to marking facial landmarks (eyes, nose, mouth points)

device = training.set_device(True)
batch_size = 128
epoch_count = 10
lr = 0.01

num_features = 40 #40 feature annotations per image
num_channels = 3

class CelebDataset(Dataset):
    def __init__(self, annotations_file, img_dir, train=True, transform=None, target_transform=None, device=training.set_device()):
        self.img_dir = img_dir

        annotation_lines = []
        with open(annotations_file) as f:
            for line in f.readlines():
                annotation_lines.append(line.split())
        self.label_names = annotation_lines[1] #names of different attributes
        if train:
            self.length = 162769 #80% train
            annotation_array = np.array(annotation_lines[2:162771])[:, 1:].astype(int) #annotation_array is 2d array of attribute values
        else:
            self.length = 39829  #20% test
            annotation_array = np.array(annotation_lines[162772:])[:, 1:].astype(int)
        annotation_array[np.where(annotation_array == -1)] = 0 #replace all -1 with 0 for attributes that are not present
        self.img_annotations = annotation_array #skips image count and label name lines, ignores image name

        self.transform = transform
        self.target_transform = target_transform
        self.device = device

    def __len__(self):
        return self.length

    def __getitem__(self,idx):
        path_index = str(idx+1).zfill(6) + '.jpg'#formats string similar to '000137.jpg'
        img_path = os.path.join(self.img_dir, path_index)
        image = read_image(img_path).to(self.device, torch.float)
        annotations = torch.as_tensor(self.img_annotations[idx]).to(self.device, torch.float) #tensor of 0 or 1 for different attributes

        if self.transform:
            if self.transform is not ToTensor():
                image = self.transform()(image)
        if self.target_transform:
            annotations = self.target_transform()(annotations)

        return image, annotations

training_data = CelebDataset(annotations_file = 'celeb_data\list_attr_celeba.txt',
                             img_dir = 'celeb_data\img_align_celeba',
                             train=True,
                             device=device)
test_data = CelebDataset(annotations_file = 'celeb_data\list_attr_celeba.txt',
                         img_dir = 'celeb_data\img_align_celeba',
                         train=False,
                         device=device)


train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

print("****DATA LOADED SUCCESSFULLY****")

celeb_model = training.image_model(num_categories=num_features, num_channels=num_channels, width=178, height=218, device=device)
loss_calc = nn.BCELoss()
optimizer = torch.optim.SGD(celeb_model.parameters(), lr=lr)

accuracies_by_epoch, accuracies_by_batch, loss_by_epoch, loss_by_batch = training.train_model(celeb_model, loss_calc, optimizer, 
                                                                                              train_dataloader, test_dataloader, 
                                                                                              epoch_count, device=device)

training.save_model(celeb_model)
training.plot_accuracy_and_loss(accuracies_by_epoch, losses_by_epoch, epoch_count)