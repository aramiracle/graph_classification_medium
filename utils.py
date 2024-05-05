import os
import random
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import networkx as nx

def draw_random_graph_samples(dataset, dataset_name, num_classes, num_samples=20, model_save_path='models/'):
    num_rows = 4
    num_cols = 5
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(5*num_cols, 5*num_rows))
    fig.suptitle('Random Samples of Graphs')

    # Define a colormap based on the number of unique classes
    cmap = plt.cm.get_cmap('tab10', num_classes)
    
    # Get random indices for sampling
    random_indices = random.sample(range(len(dataset)), num_samples)
    
    for i, index in enumerate(random_indices):
        row = i // num_cols
        col = i % num_cols

        data = dataset[index]
        edge_index, y = data.edge_index, data.y
        G = nx.Graph()
        for src, dst in edge_index.t().tolist():
            G.add_edge(src, dst)

        # Map class labels to colors
        nodes_color = cmap(y[0])
 
        pos = nx.spring_layout(G)  # Positions for all nodes
        nx.draw(G, pos, with_labels=True, node_color=nodes_color, node_size=100, edge_color='k', linewidths=1, font_size=5, ax=axs[row, col])
        axs[row, col].set_title(f'Graph Sample {index}')
    
    # Custom legend for class labels
    handles = [plt.Line2D([], [], color=cmap(i), marker='o', linestyle='', markersize=10, label=f'Class {i}') for i in range(num_classes)]
    plt.legend(handles=handles, title='Class Labels', loc='center left', bbox_to_anchor=(1, 0.5))

    # Save the plot
    plt.savefig(os.path.join(model_save_path, f'{dataset_name}/random_samples_of_graphs.png'))

def train_model(model, train_loader, test_loader, optimizer, criterion, num_epochs=1000, model_save_path='models/'):
    best_model_path = os.path.join(model_save_path, 'best_model.pth')
    best_test_accuracy = 0.0
    best_f1 = 0.0
    
    log_data = {'Epoch': [], 'Average Loss': [], 'Test Accuracy': [], 'F1 Score': []}
    
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
        true_labels = []
        predicted_labels = []
        with torch.no_grad():
            for data in test_loader:
                x, edge_index, y = data.x, data.edge_index, data.y
                out = model(x, edge_index, data.batch)
                _, predicted = torch.max(out, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
                true_labels.extend(y.numpy())
                predicted_labels.extend(predicted.numpy())

        test_accuracy = correct / total
        f1 = f1_score(true_labels, predicted_labels, average='weighted')
        tqdm.write(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.5f}, Test Accuracy: {test_accuracy:.5f}, F1 Score: {f1:.5f}')
        
        # Log data
        log_data['Epoch'].append(epoch + 1)
        log_data['Average Loss'].append(avg_loss)
        log_data['Test Accuracy'].append(test_accuracy)
        log_data['F1 Score'].append(f1)

        # Save the best model based on test accuracy and F1-score
        if test_accuracy > best_test_accuracy or (test_accuracy == best_test_accuracy and best_f1 < f1):
            tqdm.write("$$$ best model is updated according to accuracy! $$$")
            best_test_accuracy = test_accuracy
            best_f1 = f1
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