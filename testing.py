from training import set_device, image_model
import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from torchvision.transforms import ToTensor
import PIL.Image as Image
import tkinter as tk
from tkinter import filedialog as fd

default_path = 'default'

def get_image(shape):
    image_is_valid = False
    while not image_is_valid:
        image_path = fd.askopenfilename(title="Please select an input image: ")
        try:
            input_image = Image.open(image_path)
            image_is_valid = True
        except:
            print("**ERROR** Image path is invalid!")
    return input_image.resize(shape)

def set_model_state(model):
    file_is_valid = False
    if input("Use default model? (y/n)") == 'y':
        model.load_state_dict(torch.load(default_path))
        return model
    while not file_is_valid:
        file_path = fd.askopenfilename(title="Select a model state: ")
        try:
            print("Loading state dict...")
            model.load_state_dict(torch.load(file_path))
            file_is_valid = True
        except:
            print("**ERROR** Model path is invalid!")
    return model

def get_prediction(model, image, label_names, device):
    with torch.no_grad():
        print("Processing image...")
        image_tensor = image.to(device=device)
        predicted = model(image_tensor)
        indexes = np.where(torch.round(predicted).cpu().numpy() == 1)[1].astype(int)
        predictions = []
        for i in indexes:
            predictions.append(label_names[i])
    return predictions

def get_prediction_percents(model, image, device):
    with torch.no_grad():
        print("Processing image...")
        image_tensor = image.to(device, torch.float)
        predicted = model(image_tensor)
    return predicted.cpu().numpy()