# Chest X-ray Classification Project

This project aims to classify chest X-ray images into two categories: "Normal" and "Pneumonia" using deep learning techniques. The code provided here includes data preprocessing, model training, evaluation, and visualization.

## Table of Contents

- [Project Overview](#project-overview)
- [Dependencies](#dependencies)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Visualization](#visualization)
- [Conclusion](#conclusion)

## Project Overview

Chest X-ray imaging is a common diagnostic tool in the medical field. Automatic classification of chest X-ray images into normal and pneumonia cases can assist radiologists in their diagnosis and improve healthcare efficiency. In this project, we leverage deep learning techniques to develop a model capable of accurately classifying chest X-ray images.

## Dependencies

The code is implemented in Python and relies on the following libraries:
- PyTorch
- torchvision
- numpy
- matplotlib
- scikit-learn
- OpenCV

You can install the necessary dependencies using the provided `requirements.txt` file.

```bash
pip install -r requirements.txt
```

## Dataset

The dataset consists of chest X-ray images categorized into three subsets: training, validation, and test. Each subset contains images labeled as either "Normal" or "Pneumonia". The data preprocessing steps include resizing, normalization, and augmentation (for training).

## Data Preprocessing

Data preprocessing involves loading the dataset, applying transformations such as resizing and normalization, and creating custom dataset classes for efficient loading using PyTorch's DataLoader.

## Model Architecture

We utilize a pre-trained DenseNet model for feature extraction and modify its fully connected layer for binary classification. The model is trained using the Adam optimizer with a custom learning rate scheduler and class weights to handle class imbalance.

## Training

The training loop iterates over epochs, computing training and validation losses, and updating the model's parameters. Early stopping is implemented to prevent overfitting, and the best model is saved based on validation loss.

## Evaluation

Model performance is evaluated using various metrics, including accuracy, precision, recall, F1 score, and confusion matrix. We provide functions to evaluate the model on training, validation, and test subsets.

## Visualization

We visualize training and validation losses, training and validation accuracies, and generate confusion matrices for each subset. Additionally, we create visualizations of the input images along with their predicted and true labels, as well as a fly-through video showcasing the predictions.

## Conclusion

This project demonstrates the application of deep learning techniques for chest X-ray classification. The trained model achieves promising results and can be further optimized for real-world deployment in clinical settings.
