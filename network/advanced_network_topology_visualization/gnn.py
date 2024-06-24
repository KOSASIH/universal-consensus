# gnn.py
import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
import torch_geometric.data as pyg_data

class GNN(nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.conv1 = pyg_nn.GraphConv(16, 32)
        self.conv2 = pyg_nn.GraphConv(32, 64)
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 16)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = torch.relu(self.conv1(x, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def visualize_network_topology(network_data):
    # Create a PyTorch Geometric data object
    data = pyg_data.Data(x=torch.tensor(network_data.nodes), edge_index=torch.tensor(network_data.edges).t().contiguous())

    # Initialize the GNN model
    model = GNN()

    # Forward pass
    output = model(data)

    # Visualize the output using a library like Matplotlib or Plotly
    import matplotlib.pyplot as plt
    plt.scatter(output[:, 0], output[:, 1])
    plt.show()
