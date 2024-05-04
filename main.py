import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import GINConv, global_mean_pool
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

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

def train_model(model, train_loader, test_loader, optimizer, criterion, num_epochs=1000, model_save_path='models/'):
    min_loss = float('inf')
    best_model_path = os.path.join(model_save_path, 'best_model.pth')
    best_test_accuracy = 0.0
    
    log_data = {'Epoch': [], 'Average Loss': [], 'Test Accuracy': []}
    
    for epoch in tqdm(range(num_epochs), desc='Training'):
        total_loss = 0
        total_samples = 0
        
        # Training loop
        model.train()
        for data in train_loader:
            x, edge_index, y = data.x, data.edge_index, data.y
            optimizer.zero_grad()
            out = model(x, edge_index, data.batch)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * data.num_graphs
            total_samples += data.num_graphs
            
        avg_loss = total_loss / total_samples

        # Test loop
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                x, edge_index, y = data.x, data.edge_index, data.y
                out = model(x, edge_index, data.batch)
                _, predicted = torch.max(out, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

        test_accuracy = correct / total
        tqdm.write(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.5f} Test Accuracy: {test_accuracy:.5f}')
        
        # Log data
        log_data['Epoch'].append(epoch + 1)
        log_data['Average Loss'].append(avg_loss)
        log_data['Test Accuracy'].append(test_accuracy)

        # Save the best model based on test accuracy
        if test_accuracy > best_test_accuracy or (test_accuracy == best_test_accuracy and min_loss > avg_loss):
            tqdm.write("$$$ Best model is updated! $$$")
            best_test_accuracy = test_accuracy
            min_loss = avg_loss
            os.makedirs(model_save_path, exist_ok=True)
            torch.save(model.state_dict(), best_model_path)

    # Load the best model for evaluation
    model.load_state_dict(torch.load(best_model_path))
    
    # Convert log_data to DataFrame
    log_df = pd.DataFrame(log_data)
    log_df.to_csv(os.path.join(model_save_path, 'train_log.csv'), index=False)


def extract_embeddings(model, dataset):
    model.eval()
    embeddings_list = []
    for data in dataset:
        x, edge_index = data.x, data.edge_index
        with torch.no_grad():
            embedding = model.extract_embedding(x, edge_index, data.batch)
        embeddings_list.append(embedding.numpy())
    embeddings_all = np.concatenate(embeddings_list)
    return embeddings_all

def predict_labels(model, test_loader):
    predicted_labels = []
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            x, edge_index, _ = data.x, data.edge_index, data.batch
            out = model(x, edge_index, data.batch)
            _, predicted = torch.max(out, 1)
            predicted_labels.extend(predicted.numpy())
    return predicted_labels

def visualize_embeddings(embeddings, true_labels, predicted_labels, model_save_path):
    tsne = TSNE(n_components=2)
    embeddings_tsne = tsne.fit_transform(embeddings)

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle('t-SNE Visualization of Embeddings')

    for ax, labels, title in zip(axs, [true_labels, predicted_labels], ['True Labels', 'Predicted Labels']):
        for label in np.unique(labels):
            indices = np.where(labels == label)
            ax.scatter(embeddings_tsne[indices, 0], embeddings_tsne[indices, 1], label=f'Class {label}')
        ax.set_title(title)
        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        ax.legend()
        ax.grid(True)

    plt.savefig(os.path.join(model_save_path, 'test_embedding_with_labels.png'))

def main():
    batch_size = 64
    num_epochs = 1000

    datasets = ['MUTAG', 'ENZYMES', 'PROTEINS']

    for dataset_name in datasets:
        dataset = TUDataset(root=f'/tmp/{dataset_name}', name=dataset_name, use_node_attr=True)
        os.makedirs(f'models/{dataset_name}', exist_ok=True)
        
        # Split dataset into training and test sets
        train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, shuffle=False)

        model = GINClassification(in_channels=dataset.num_node_features, hidden_channels=1000, out_channels=100, num_classes=dataset.num_classes)

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        train_model(model, train_loader, test_loader, optimizer, criterion, num_epochs=num_epochs, model_save_path=f'models/{dataset_name}')

        # Extract embeddings for test set
        test_embeddings = extract_embeddings(model, test_loader)

        # Get true labels for test set
        true_labels = []
        for data in test_loader:
            true_labels.append(data.y.numpy())
        true_labels = np.concatenate(true_labels)

        # Predict labels for test set
        predicted_labels = predict_labels(model, test_loader)

        # Plot t-SNE visualization with true and predicted labels
        visualize_embeddings(test_embeddings, true_labels, predicted_labels, model_save_path=f'models/{dataset_name}')

if __name__ == "__main__":
    main()
