import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
import pandas as pd
from PIL import Image
import numpy as np

class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        in_features = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.resnet18(x)

class ShipDataset(Dataset):
    def __init__(self, dataframe, image_path, transform=None, process_images=True):
        self.dataframe = dataframe
        self.image_path = image_path
        self.transform = transform
        self.process_images = process_images
    
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_path, self.dataframe.iloc[idx, 0])
        original_image = cv2.imread(img_name)
        # original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        if self.process_images:
            processed_image = cv2.detailEnhance(original_image, sigma_s=10, sigma_r=0.15)
            image = Image.fromarray(processed_image)
        else:
            image = Image.fromarray(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))

        label = int(self.dataframe.iloc[idx, 1]) - 1

        if self.transform:
            image = self.transform(image)

        return image, label

def load_data(data_path, image_path, sample_frac, process_images=True):
    df = pd.read_csv(data_path)
    df = df.sample(frac=sample_frac, random_state=42) 

    transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
    ])

    dataset = ShipDataset(df, image_path, transform=transform, process_images=process_images)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    return loader


def train_ship_classifier(model_path, sample_frac=1, process_images=True):
    data_path = os.path.join("./train/", "train.csv")
    image_folder = os.path.join("./train/", "train/")

    train_loader = load_data(data_path, image_folder, sample_frac, process_images)

    model = ResNet18(num_classes=5)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(10):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        scheduler.step()
        print(f"Epoch {epoch + 1}/{10}")

    torch.save(model.state_dict(), model_path)
