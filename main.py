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
using_previous_model = False
if input("Use previous model? (y/n)") == 'y':
    using_previous_model = True
device = training.set_device(True)
batch_size = 64
epoch_count = 20
lr = 1e-1

num_features = 40 #40 feature annotations per image
num_channels = 3
image_shape = (178,218)
label_names = ["5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes", 
                "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair", "Blurry", 
                "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", "Goatee", 
                "Gray_Hair", "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open", 
                "Mustache", "Narrow_Eyes", "No_Beard", "Oval_Face", "Pale_Skin", "Pointy_Nose", 
                "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair", 
                "Wavy_Hair", "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick", 
                "Wearing_Necklace", "Wearing_Necktie", "Young"]

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
        image = read_image(img_path).to(torch.float)
        annotations = torch.as_tensor(self.img_annotations[idx]).to(torch.float) #tensor of 0 or 1 for different attributes

        if self.transform:
            if self.transform is not ToTensor():
                image = self.transform()(image)
        if self.target_transform:
            annotations = self.target_transform()(annotations)
        return image, annotations

if not using_previous_model: #load dataset if training
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

celeb_model = training.image_model(num_categories=num_features, num_channels=num_channels, width=image_shape[0], height=image_shape[1], device=device)
loss_calc = nn.BCELoss()
optimizer = torch.optim.SGD(celeb_model.parameters(), lr=lr)

if using_previous_model:
    image = ToTensor()(testing.get_image(image_shape)) #get image tensor
    image_4d = image[None,...] #reshape to 4d input, since model is trained on batches this is like a batch of size 1
    celeb_model = testing.set_model_state(celeb_model) #retrieve saved model state
    if input("Use percent confidence? (y/n)") == 'y':
        use_percents = True
    else:
        use_percents = False
    ToPILImage()(image).show()
    if use_percents: 
        percents = testing.get_prediction_percents(celeb_model, image_4d, device)[0] #run through model and get array of percents for each attribute
        attrs = list(zip(label_names, percents))#zip labels with percents into list of tuples
        sorted_attrs = sorted(attrs, key = lambda x: x[1], reverse=True) #sort percents in decending order
        print("\n****PREDICTIONS****\n")
        for attr, percent in sorted_attrs:
            print(f"{attr}: {(percent*100):.2f}%")
    else:
        attrs = testing.get_prediction(celeb_model, image_4d, label_names, device)
        print("\n****PREDICTIONS****\n")
        for attr in attrs:
            print(f"{attr}")
else:
    accuracies_by_epoch, accuracies_by_batch, loss_by_epoch, loss_by_batch = training.train_model(celeb_model, loss_calc, optimizer, 
                                                                                                train_dataloader, test_dataloader, 
                                                                                                epoch_count, device=device)
    training.save_model(celeb_model)
    training.plot_accuracy_and_loss(accuracies_by_epoch, loss_by_epoch, epoch_count)