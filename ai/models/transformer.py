import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, dropout):
        super(TransformerModel, self).__init__()
        self.encoder = TransformerEncoder(input_dim, num_heads, dropout)
        self.decoder = TransformerDecoder(output_dim, num_heads, dropout)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, num_heads, dropout):
        super(TransformerEncoder, self).__init__()
        self.self_attn = MultiHeadAttention(input_dim, num_heads, dropout)
        self.feed_forward = FeedForward(input_dim, dropout)

    def forward(self, x):
        x = self.self_attn(x)
        x = self.feed_forward(x)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, output_dim, num_heads, dropout):
        super(TransformerDecoder, self).__init__()
        self.self_attn = MultiHeadAttention(output_dim, num_heads, dropout)
        self.feed_forward = FeedForward(output_dim, dropout)

    def forward(self, x):
        x = self.self_attn(x)
        x = self.feed_forward(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads, dropout):
        super(MultiHeadAttention, self).__init__()
        self.query_linear = nn.Linear(input_dim, input_dim)
        self.key_linear = nn.Linear(input_dim, input_dim)
        self.value_linear = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        queries = self.query_linear(x)
        keys = self.key_linear(x)
        values = self.value_linear(x)
        attention_weights = torch.matmul(queries, keys.T) / math.sqrt(input_dim)
        attention_weights = self.dropout(attention_weights)
        context = torch.matmul(attention_weights, values)
        return context

class FeedForward(nn.Module):
    def __init__(self, input_dim, dropout):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class TransformerDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        return data, label

def train(model, device, loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, labels) in enumerate(loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item()}')

def test(model, device, loader):
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
