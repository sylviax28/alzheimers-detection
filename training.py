#!/usr/bin/env python3

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.optim import SGD
from tqdm import tqdm
import pandas as pd
from PIL import Image

#new comment

class CSVDataset(Dataset):
    
    label_map = {
        'noimpairment': 0,
        'impairment': 1
    }

    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
    
    # number of rows
    def __len__(self):
        return len(self.data)
    
    # get row at an index
    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]
        label = self.data.iloc[idx, 1]
        
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        label = CSVDataset.label_map[label]
        return image, int(label)

transform = transforms.Compose([
    transforms.ToTensor()
])
    
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean
        std=[0.229, 0.224, 0.225]    # ImageNet std
    )
])

train_dataset = CSVDataset("train.csv", transform=transform)
val_dataset   = CSVDataset("val.csv", transform=transform)
test_dataset  = CSVDataset("test.csv", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
class CNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CNN, self).__init__()
        # first convolutional layer
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, padding=1)
        # pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7,7))  # shrink to 7x7
        # second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(16*7*7, num_classes)        # match adaptive pool output
    
    # defines a sequence of operations applied to an input x as it passes through the CNN
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x))) 
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.dropout(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN(in_channels=3, num_classes=2).to(device)
epochs = 30

def train_model(train_dl, val_dl, model, epochs=10):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=1e-4, momentum=0.9)
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for data, targets in tqdm(train_dl):
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            scores = model(data)
            loss = criterion(scores, targets)
            running_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            
            predicted = scores.argmax(dim=1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        train_loss = running_loss / len(train_dl)
        train_acc = 100 * correct / total

        # Validation
        model.eval()
        with torch.no_grad():
            correct_val = 0
            total_val = 0
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                predicted = outputs.argmax(dim=1)
                total_val += y.size(0)
                correct_val += (predicted == y).sum().item()
            val_acc = 100 * correct_val / total_val

        print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}, "
              f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

train_model(train_loader, val_loader, model, epochs=30)
torch.save(model.state_dict(), "alz_model.pth")