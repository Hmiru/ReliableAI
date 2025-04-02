import torch
import torch.nn as nn
import torch.nn.functional as F
from config import device
class MnistConvNet(nn.Module):
    """
    Mnist 데이터셋을 위한 간단한 CNN 모델
    """
    def __init__(self, num_classes=10):
        super(MnistConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc2 = nn.Linear(8*14*14, num_classes) 
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x))) 
        x = x.view(x.size(0), -1) # flatten
        x = self.fc2(x)
        return x

class CifarConvNet(nn.Module):
    """
    cifar-10 데이터셋을 위한 간단한 CNN 모델
    """
    def __init__(self, num_classes=10):
        super(CifarConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64*8*8, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x))) # 32*32->16*16
        x= self.pool(self.relu(self.conv2(x))) # 16*16->8*8
        x = x.view(x.size(0), -1) # flatten
        x=self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
    
def train(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            # 모델 예측 및 손실 계산
            outputs = model(images)
            loss = criterion(outputs, labels)
            # 기울기 계산
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the model on the test images: {100 * correct / total:.2f}%')

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.eval()
    return model
