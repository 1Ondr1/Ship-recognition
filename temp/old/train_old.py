import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from PIL import Image
import pandas as pd

# ship = {'Cargo': 1, 
#         'Military': 2, 
#         'Carrier': 3, 
#         'Cruise': 4, 
#         'Tankers': 5}

class ShipClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ShipClassifier, self).__init__()
        self.features = models.resnet18(pretrained=True)
        in_features = self.features.fc.in_features
        self.features.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.features(x)

class ShipDataset(Dataset):
    def __init__(self, dataframe, image_path, transform=None):
        self.dataframe = dataframe
        self.image_path = image_path
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_path, self.dataframe.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")
        label = int(self.dataframe.iloc[idx, 1]) - 1

        if self.transform:
            image = self.transform(image)

        return image, label

def load_data(data_path, image_path, sample_frac=1.0):
    df = pd.read_csv(data_path)
    df = df.sample(frac=sample_frac, random_state=42)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    return train_df, test_df

def build_model(num_classes=5):
    model = ShipClassifier(num_classes=num_classes)

    return model

def train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs=10, device="cpu"):
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

    return model

if __name__ == "__main__":
    data_path = "./train/train.csv"
    image_path = "./train/images/"
    sample_frac = 1

    train_df, test_df = load_data(data_path, image_path, sample_frac)

    transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
    ])

    train_dataset = ShipDataset(train_df, image_path, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = build_model(num_classes=5)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    trained_model = train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs=10, device="cpu")

    # Збереження моделі
    torch.save(trained_model.state_dict(), "ship_classification_model.pth")
