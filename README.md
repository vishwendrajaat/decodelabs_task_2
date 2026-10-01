# Project 2 — Data Classification Using AI

## Iris Flower Classification using K-Nearest Neighbors

This project is part of the DecodeLabs AI Industrial Training Program.

The project demonstrates how supervised machine learning can be used to classify data into different categories. The Iris dataset is used, and a K-Nearest Neighbors (KNN) classification algorithm is applied.

## Project Objective

The objective of this project is to build a basic AI classification model that can:

- Load a dataset
- Separate features and target values
- Split the dataset into training and testing data
- Apply feature scaling
- Train a K-Nearest Neighbors classification model
- Make predictions
- Evaluate the model using classification metrics
- Predict the class of a new Iris flower

## Dataset

The project uses the built-in **Iris dataset** provided by scikit-learn.

The dataset contains:

- 150 samples
- 4 input features
- 3 flower classes

### Features

1. Sepal Length
2. Sepal Width
3. Petal Length
4. Petal Width

### Classes

- Setosa
- Versicolor
- Virginica

## Machine Learning Algorithm

The project uses **K-Nearest Neighbors (KNN)**.

The value of K is set to:

```text
K = 5
```

KNN classifies a new data point based on the classes of its nearest neighboring data points.

## Project Workflow

```text
Load Iris Dataset
       ↓
Separate Features and Target
       ↓
80% Training / 20% Testing Split
       ↓
Feature Scaling using StandardScaler
       ↓
K-Nearest Neighbors (K=5)
       ↓
Train Model
       ↓
Make Predictions
       ↓
Evaluate Model
       ↓
Predict New Flower
```

## Technologies Used

- Python
- Scikit-learn
- Machine Learning
- Supervised Learning
- K-Nearest Neighbors

## Libraries Used

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
```

## Requirements

Make sure Python is installed on your system.

Install scikit-learn using:

```bash
pip install scikit-learn
```

## How to Run

Open the project folder in VS Code.

Open the terminal and run:

```bash
python "Data Classification Using AI"
```

If your Python installation requires the `py` command, you can use:

```bash
py "Data Classification Using AI"
```

## Model Evaluation

The model evaluates its predictions using:

- Accuracy
- F1 Score
- Confusion Matrix
- Classification Report

The classification report also provides precision and recall for the different flower classes.

## New Flower Prediction

The program also accepts a sample flower measurement and uses the trained KNN model to predict its class.

Example input:

```text
5.1
3.5
1.4
0.2
```

The model then predicts the corresponding Iris flower class.

## Project Structure

```text
Project 2/
│
├── Data Classification Using AI
└── README.md
```

## Learning Outcomes

Through this project, the following concepts are demonstrated:

- Dataset loading
- Feature and target separation
- Train-test splitting
- Feature scaling
- Supervised machine learning
- KNN classification
- Model training
- Prediction
- Model evaluation
- Confusion matrix
- F1 score
- Basic machine learning workflow

## Project Status

**Completed**

Project: **Project 2 — Data Classification Using AI**

Algorithm: **K-Nearest Neighbors (KNN)**

Dataset: **Iris Dataset**
