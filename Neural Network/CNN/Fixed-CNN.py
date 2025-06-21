# A fixed CNN (Convolutional Neural Network)

import gdown
import os
import pickle
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt


# Download and unzip data
url = "https://drive.google.com/uc?id=1YQiwQ2XmCNQoQv5G898cngZYuv00gW47"
output = "mal.zip"
gdown.download(url, output, quiet=False)

os.system("unzip -o mal.zip")
os.system("rm mal_data/train/.DS_Store")
os.system("rm mal_data/test/.DS_Store")


# Helper function to find classes
def find_classes(directory: str):
    """Finds the class folders in a dataset."""
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


_, class_to_idx = find_classes("mal_data/train")


# Custom Dataset class
class MalwareDataset(Dataset):
    def __init__(self, data_path, input_length=4096, transform=False):
        self.data_path = data_path
        self.input_length = input_length
        _, self.class_to_idx = find_classes(data_path)
        self.malware_files = []

        for target_class in sorted(self.class_to_idx.keys()):
            class_index = self.class_to_idx[target_class]
            target_dir = os.path.join(data_path, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    item = path, class_index
                    self.malware_files.append(item)

    def __getitem__(self, index):
        item = self.malware_files[index]
        with open(item[0], "rb") as f:
            tmp = [i + 1 for i in f.read()[:self.input_length]]
            tmp = tmp + [0] * (self.input_length - len(tmp))
        return np.array(tmp), item[1]

    def __len__(self):
        return len(self.malware_files)


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.classifier = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=5, stride=2),  # Conv1
            nn.MaxPool1d(kernel_size=2, stride=2),  # Pool1
            nn.Conv1d(4, 8, kernel_size=3, stride=2),  # Conv2
            nn.MaxPool1d(kernel_size=2, stride=2),  # Pool2
            nn.Conv1d(8, 16, kernel_size=3, stride=2),  # Conv3
            nn.MaxPool1d(kernel_size=2, stride=2),  # Pool3
            nn.Conv1d(16, 32, kernel_size=3, stride=1),  # Conv4
            nn.AdaptiveMaxPool1d(1),  # Ensures output size is fixed regardless of input size
            nn.Flatten(),
            nn.Linear(32, 64),  # Adjust Linear input size
            nn.ReLU(),
            nn.Linear(64, 6)
        )

    def forward(self, x):
        return self.classifier(x)



# Model setup
model = ConvNet().cuda()
model.train()

# Hyperparameters
lr = 0.001
num_epochs = 10
batch_size = 16
input_len = 256  # Length of input sequence

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Dataset and DataLoader setup
train_data = MalwareDataset("mal_data/train", input_length=input_len)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_data = MalwareDataset("mal_data/test", input_length=input_len)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Lists to track loss and accuracy
loss_list = []
acc_list = []

# Training loop
for epoch in range(num_epochs):
    for i, (samples, labels) in enumerate(train_loader):
        samples = samples.type(torch.FloatTensor).cuda()  # Convert to float and move to GPU
        labels = labels.cuda()  # Move labels to GPU

        # Forward pass
        outputs = model(samples.unsqueeze(1))  # Add channel dimension

        # Compute loss
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track accuracy
        _, predicted = torch.max(outputs.data, 1)  # Get predictions
        correct = (predicted == labels).sum().item()
        acc_list.append(correct / labels.size(0))

        # Print loss and accuracy every 200 steps
        if i % 200 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(train_loader)}], '
                  f'Loss: {loss.item():.4f}, Accuracy: {(correct / labels.size(0)) * 100:.2f}%')

# Model evaluation on test set
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for samples, labels in test_loader:
        samples = samples.type(torch.FloatTensor).cuda()
        labels = labels.cuda()

        outputs = model(samples.unsqueeze(1))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the model on the test samples: {:.2f}%'.format((correct / total) * 100))

# Plot accuracy and loss
plt.figure()
plt.plot(acc_list)
plt.title('Accuracy')

plt.figure()
plt.plot(loss_list)
plt.title('Loss')

plt.show()
