from training import set_device, image_model, image_grid
import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
from torchvision.transforms import ToTensor
import PIL.Image as Image
import tkinter as tk
from tkinter import filedialog as fd

root = tk.Tk()
root.title("")
root.resizable(False, False)
root.geometry('300x150')

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
    while not file_is_valid:
        file_path = fd.askopenfilename(title="Select a model state: ")
        try:
            model.load_state_dict(torch.load(file_path))
            file_is_valid = True
        except:
            print("**ERROR** Model path is invalid!")
    return model

def get_prediction(model, image, labels):
    image_tensor = ToTensor()(image)
    predicted = model(image_tensor)
    return labels[torch.argmax(predicted)]