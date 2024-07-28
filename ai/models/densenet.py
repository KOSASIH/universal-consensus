import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class DenseNetModel(nn.Module):
    def __init__(self, num_classes):
        super(DenseNetModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.dense1 = self._make_dense_layer(64, 128, 6)
        self.transition1 = self._make_transition_layer(128, 256)
        self.dense2 = self._make_dense_layer(256, 512, 12)
        self.transition2 = self._make_transition_layer(512, 1024)
        self.dense3 = self._make_dense_layer(1024, 2048, 24)
        self.transition3 = self._make_transition_layer(2048, 4096)
        self.fc = nn.Linear(4096, num_classes)

    def _make_dense_layer(self, inplanes, planes, blocks):
        layers = []
        for i in range(blocks):
            layers.append(DenseBlock(inplanes, planes))
            inplanes += planes
        return nn.Sequential(*layers)

    def _make_transition_layer(self, inplanes, planes):
        return nn.Sequential(
            nn.BatchNorm2d(inplanes),
            nn.ReLU(),
            nn.Conv2d(inplanes, planes, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.dense1(x)
        x = self.transition1(x)
        x = self.dense2(x)
        x = self.transition2(x)
        x = self.dense3(x)
        x = self.transition3(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class DenseBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
                out = torch.cat([x, out], 1)
        return out

def load_model(model_name, num_classes):
    model = DenseNetModel(num_classes)
    if model_name == "densenet121":
        model.load_state_dict(torch.load("densenet121.pth"))
    elif model_name == "densenet169":
        model.load_state_dict(torch.load("densenet169.pth"))
    elif model_name == "densenet201":
        model.load_state_dict(torch.load("densenet201.pth"))
    return model

def train_model(model, device, loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, labels) in enumerate(loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item()}')

def test_model(model, device, loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, labels in loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()

    accuracy = correct / len(loader.dataset)
    print(f'Test Loss: {test_loss / len(loader)}')
    print(f'Test Accuracy: {accuracy:.2f}%')

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_classes = 10
    model_name = "densenet121"

    # Load the dataset
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Load the model
    model = load_model(model_name, num_classes)

    # Define the optimizer and scheduler
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Train the model
    for epoch in range(100):
        train_model(model, device, train_loader, optimizer, epoch)
        scheduler.step()

    # Test the model
    test_model(model, device, test_loader)

if __name__ == '__main__':
    main()
