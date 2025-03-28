import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import os
import random
from PIL import Image

# Define Siamese Neural Network
class SiameseNet(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = resnet50(pretrained=True)

        for param in resnet.parameters():
            param.requires_grad = False

        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, img1, img2):
        feature1 = self.feature_extractor(img1).view(img1.size(0), -1)
        feature2 = self.feature_extractor(img2).view(img2.size(0), -1)

        difference = torch.abs(feature1 - feature2)
        result = self.fc(difference)

        return result

# Dataset
class CustomDataSet(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.image_paths = os.listdir(self.root)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        folder = os.path.join(self.root, self.image_paths[index])

        img1 = Image.open(os.path.join(folder, random.choice(os.listdir(folder)))).convert('RGB')
        img2 = Image.open(os.path.join(folder, random.choice(os.listdir(folder)))).convert('RGB')
        label = 1

        if random.random() > 0.5:
            label = 0
            different_folder = random.choice(self.image_paths)
            while different_folder == self.image_paths[index]:  # Ensure different class
                different_folder = random.choice(self.image_paths)
            img2 = Image.open(os.path.join(self.root, different_folder, random.choice(os.listdir(os.path.join(self.root, different_folder))))).convert('RGB')

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, torch.tensor([label], dtype=torch.float32)  # Keep correct shape

# Hyperparameters
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEFAULT_DIM_SIZE = 224

# Correct normalization
transform = transforms.Compose([
    transforms.Resize((DEFAULT_DIM_SIZE, DEFAULT_DIM_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
"""
train_dataset = CustomDataSet('datasets/logos/train/', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

SNN_MODEL = SiameseNet().to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(SNN_MODEL.parameters(), lr=0.0001)  # Lower LR
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)  # Halve LR every 10 epochs

for name, param in SNN_MODEL.feature_extractor.named_parameters():
    if "layer4" in name:  # Unfreeze the last ResNet block
        param.requires_grad = True

num_epochs = 100
for epoch in range(num_epochs):
    SNN_MODEL.train()
    total_loss = 0

    for img1, img2, label in train_loader:
        img1 = img1.to(device)
        img2 = img2.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        result = SNN_MODEL(img1, img2)
        loss = criterion(result, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    scheduler.step()
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader)}")

torch.save(SNN_MODEL.state_dict(), 'siamese_model.pth')
"""

