import os
import sys
import json
import math
import torch
import random
import numpy as np
from PIL import Image
import torch.nn as nn
from typing import Optional
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch.utils.checkpoint as checkpoint

# Check gradient clipping to avoid exploding gradients
def clip_gradients(model, max_norm=1.0):
    """Clips gradients of model parameters to avoid exploding gradients."""
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-2):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def get_image_paths_and_labels(root_dir):
    image_paths = []
    labels = []
    
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith((".jpg", ".JPG", ".png", ".PNG")):
                image_paths.append(os.path.join(subdir, file))
                labels.append(os.path.basename(subdir))
    
    return image_paths, labels


class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        
        self.image_paths, self.labels = get_image_paths_and_labels(root_dir)
        self.label_to_idx = {label: idx for idx, label in enumerate(set(self.labels))}

        if transform == "train":
            self.transform =transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        if transform == "validate" :
            self.transform = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        label_idx = self.label_to_idx[label]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label_idx
    
    @staticmethod
    def collate_fn(batch):
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels



def load_image(image_path, img_size=224):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file {image_path} not found.")
    
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize(256),  # Resize the smaller side to 256
        transforms.CenterCrop(img_size),  # Crop the center to a fixed size (e.g., 224x224)
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize the pixel values to [-1, 1]
    ])
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension: (1, 3, H, W)
    return image_tensor


# Helper function to display image tensors
def show_image(original, convolved):
    original_images = tensor_to_rgb_image(original)
    convolved_images = tensor_to_rgb_image(convolved)
    
    for i, ( conv_img) in enumerate(zip( convolved_images)):
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        # plt.imshow(orig_img)
        plt.title("Original Image") 
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(conv_img)
        plt.title("Convolved Image")
        plt.axis('off')
        
        plt.show()


def tensor_to_rgb_image(tensor):
    # Assuming the tensor is in the shape (N, C, H, W) and in the range [0, 1]
    tensor = tensor.detach().cpu()
    tensor = torch.clamp(tensor, 0, 1)
    np_images = []
    for img in tensor:
        np_image = img.permute(1, 2, 0).numpy()  # Convert to (H, W, C)
        np_image = (np_image * 255).astype(np.uint8)  # Convert to RGB format
        np_images.append(np_image)
    return np_images
    
def visualize_and_save_images(original, convolved, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(original)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    axes[1].imshow(convolved)
    axes[1].set_title('Convolved Image (RGB)')
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()
    os.makedirs(output_dir, exist_ok=True)

def tensor_to_channel_images(tensor):
    # Assuming the tensor is in the shape (N, C, H, W)
    tensor = tensor.detach().cpu()
    tensor = torch.clamp(tensor, 0, 1)
    np_images = []
    for i in range(tensor.shape[1]):  # Iterate over channels
        channel_img = tensor[0, i, :, :]  # Extract the i-th channel
        np_image = channel_img.numpy()  # Convert to NumPy array
        np_image = (np_image * 255).astype(np.uint8)  # Scale to [0, 255]
        np_images.append(np_image)
    return np_images


def visualize_channel_images(channel_images):
    num_channels = len(channel_images)
    plt.figure(figsize=(15, 15))
    for i, img in enumerate(channel_images):
        plt.subplot(10, 10, i + 1)  # Adjust the grid size as needed
        plt.imshow(img, cmap='gray')
        plt.title(f"Channel {i}")
        plt.axis('off')
    plt.show()
    

def get_params_groups(model: torch.nn.Module, weight_decay: float = 1e-5):
    parameter_group_vars = {"decay": {"params": [], "weight_decay": weight_decay},
                            "no_decay": {"params": [], "weight_decay": 0.}}

    parameter_group_names = {"decay": {"params": [], "weight_decay": weight_decay},
                             "no_decay": {"params": [], "weight_decay": 0.}}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        if len(param.shape) == 1 or name.endswith(".bias"):
            group_name = "no_decay"
        else:
            group_name = "decay"

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)

    # print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())