import torch
import torch.nn as nn
import torch.optim as optim

# A simple CNN model with two convolutional layers and two fully connected layers
# Used for image classification tasks like MNIST digit recognition
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    # Defines the forward pass through the network using ReLU activations and max pooling
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
# FGSM Implementation

    # Generates adversarial examples using the Fast Gradient Sign Method (FGSM)
# by perturbing input images in the direction of the gradient
def fgsm_attack(model, data, target, epsilon):
    data.requires_grad = True
    output = model(data)
    loss = nn.CrossEntropyLoss()(output, target)
    model.zero_grad()
    loss.backward()
    data_grad = data.grad.data
    perturbed_data = data + epsilon * data_grad.sign()
    return torch.clamp(perturbed_data, 0, 1)

# Evaluation

# Evaluates model accuracy on adversarial examples generated with FGSM
# Returns the percentage of correctly classified adversarial samples
def evaluate(model, test_loader, epsilon):
    correct = 0
    total = 0
    for data, target in test_loader:
        perturbed_data = fgsm_attack(model, data, target, epsilon)
        output = model(perturbed_data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    accuracy = 100 * correct / total
    return accuracy

# Visuals

import matplotlib.pyplot as plt

# Displays the original and adversarial images side-by-side for visual comparison
def visualize_images(original, adversarial):
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(original.squeeze(), cmap='gray')
    ax[0].set_title('Original Image')
    ax[1].imshow(adversarial.squeeze(), cmap='gray')
    ax[1].set_title('Adversarial Image')
    plt.show()
