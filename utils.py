import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

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

def collect_labels(loader):
    # Get true labels for test set
    true_labels = []
    for data in loader:
        true_labels.append(data.y.numpy())
    true_labels = np.concatenate(true_labels)
    return true_labels

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