# Graph Neural Network for Graph Classification

This project implements a Graph Neural Network (GNN) for graph classification tasks using PyTorch Geometric. The GNN model is based on Graph Isomorphism Network (GIN) convolutional layers. Additionally, it includes utility functions for training, evaluation, and visualization of the model's performance on graph datasets.

## Table of Contents
- [Introduction](#introduction)
- [Dependencies](#dependencies)
- [Project Structure](#project-structure)
- [Model](#model)
- [Utilities](#utilities)
- [Usage](#usage)
- [Medium](#medium)

## Introduction

Graph Neural Networks (GNNs) are a class of neural networks designed to operate directly on graph-structured data. They enable the modeling of complex relationships and structures present in graphs, making them suitable for tasks such as graph classification. In this project, we focus on graph classification, where the goal is to predict a label for each graph in a dataset.

The model architecture utilized in this project is based on Graph Isomorphism Network (GIN) convolutional layers. GINConv layers are employed to perform message passing over the graph data, enabling the model to capture hierarchical representations of the input graphs.

## Dependencies

To run this project, you need the following dependencies:

- Python 3.x
- PyTorch
- PyTorch Geometric
- NumPy
- pandas
- scikit-learn
- tqdm
- matplotlib
- networkx

You can install these dependencies using pip:

```bash
pip install -r requirements.txt
```

for installing torch-geometric it is recommanded to see authers [website](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html). 

## Project Structure

The project is structured as follows:

- **model.py:** Defines the GINClassification model using PyTorch Geometric.
- **utils.py:** Contains utility functions for training, evaluation, and visualization.
- **main.py:** Main script to run experiments on graph datasets.
- **models/:** Directory to save trained models and training logs.
- **README.md:** Documentation file.

## Model

The model.py file contains the definition of the GINClassification model. Here's an overview of its components:

- **GINConvolution Layer:** Utilizes GINConv from PyTorch Geometric for message passing over graph data.
- **make_convolution Function:** Defines a sequence of linear layers followed by batch normalization and ReLU activation to create GINConv layers.
- **GINClassification Class:** Defines the GIN model for graph classification tasks. It consists of multiple GINConv layers followed by a linear classifier.

## Utilities

The utils.py file includes utility functions for training, evaluation, and visualization. Here's a summary of these functions:

- **train_model Function:** Trains the GNN model, evaluates performance, and saves the best model based on test accuracy.
- **extract_embeddings Function:** Extracts embeddings from the trained model for visualization or further analysis.
- **predict_labels Function:** Predicts labels for a given dataset using the trained model.
- **collect_labels Function:** Collects true labels from a given dataset for evaluation.
- **visualize_embeddings Function:** Visualizes embeddings using t-SNE for dimensionality reduction and comparison of true and predicted labels.

## Usage

To use the project, follow these steps:

Clone the repository:

```bash
git clone https://github.com/yourusername/your-repository.git
cd your-repository
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the main script:

```bash
python main.py
```

This will train the GNN model on specified datasets, evaluate its performance, and visualize embeddings using t-SNE.
Results

After running the main script, you can find trained models, training logs, and visualization results in the models/ directory. Training logs are saved as CSV files, and visualization results are saved as PNG images.

## Medium

dfs