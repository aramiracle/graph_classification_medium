import os
import torch.nn as nn
import torch.optim as optim
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from sklearn.model_selection import train_test_split

from model import GINClassification
from utils import train_model, extract_embeddings, collect_labels, predict_labels, visualize_embeddings


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

        # Collect true labels
        true_labels = collect_labels(test_loader)

        # Predict labels for test set
        predicted_labels = predict_labels(model, test_loader)

        # Plot t-SNE visualization with true and predicted labels
        visualize_embeddings(test_embeddings, true_labels, predicted_labels, model_save_path=f'models/{dataset_name}')

if __name__ == "__main__":
    main()
