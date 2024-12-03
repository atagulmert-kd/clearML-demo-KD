import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
from clearml import Task, Logger
from tqdm import tqdm

# Initialize the task
task = Task.init(project_name="Demo Project", task_name="MNIST Training Demo", task_type=Task.TaskTypes.training)

# Data Preparation
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)

# Model Definition

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
model = SimpleCNN()

# Loss and Optimizer

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Log parameters to ClearML
task.connect({"learning_rate": 0.001, "batch_size": 64, "epochs": 5})

# Train Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

epochs = 5
train_losses = []
test_accuracies = []

for epoch in range(epochs):
    model.train()
    epoch_loss = 0

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch")

    for batch_idx, (data,target) in enumerate(progress_bar):
        data, target = data.to(device), target.to(device)

        #forward
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        epoch_loss += loss.item()

        #backward
        loss.backward()
        optimizer.step()

        progress_bar.set_postfix({"Batch Loss": loss.item()})

    train_losses.append(epoch_loss/len(train_loader))
    task.get_logger().report_scalar(title="Loss",series="Train",value=epoch_loss/len(train_loader), iteration=epoch)

    # Model Evaluation
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    accuracy = correct / total
    test_accuracies.append(accuracy)
    task.get_logger().report_scalar(title="Accuracy",series="Test",value=accuracy, iteration=epoch)

    # Confusion Matrix
    if epoch == epochs - 1:
        cm = confusion_matrix(all_targets, all_preds)
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.savefig("confusion_matrix.png")
        task.get_logger().report_image(title="Confusion Matrix", series="Test", iteration=epoch, local_path="confusion_matrix.png")

# Save Model
torch.save(model.state_dict(), "mnist_model.pth")
task.upload_artifact("Model Weights", artifact_object="mnist_model.pth")

# Log Training Graphs
plt.figure()
plt.plot(range(epochs), train_losses, label="Train Loss")
plt.plot(range(epochs), test_accuracies, label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Metric")
plt.legend()
plt.savefig("training_graph.png")
task.get_logger().report_image(title="Training Metrics", series="Loss and Accuracy", iteration=epoch, local_path="training_graph.png")
