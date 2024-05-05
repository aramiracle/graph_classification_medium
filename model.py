import torch.nn as nn
from torch_geometric.nn import GINConv, global_mean_pool

def make_convolution(in_channels, out_channels):
    return GINConv(nn.Sequential(
        nn.Linear(in_channels, out_channels),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(),
        nn.Linear(out_channels, out_channels),
        nn.BatchNorm1d(out_channels),
        nn.ReLU()
    ))

class GINClassification(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_classes):
        super(GINClassification, self).__init__()
        self.conv1 = make_convolution(in_channels, hidden_channels)
        self.conv2 = make_convolution(hidden_channels, hidden_channels)
        self.conv3 = make_convolution(hidden_channels, out_channels)
        self.classifier = nn.Linear(out_channels, num_classes)


    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch=batch)
        return self.classifier(x)

    def extract_embedding(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch=batch)
        return x