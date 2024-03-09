# Iris Species Classification Project

## Overview
This project aims to develop a machine learning model to classify iris flowers into different species based on their sepal and petal characteristics. The dataset used for this project is the famous Iris dataset.

## Dataset
The dataset (`iris.data`) is loaded using Pandas, containing information about sepal length, sepal width, petal length, petal width, and the species of the iris flowers.

### Exploratory Data Analysis (EDA)
- Displaying basic information about the dataset using `df_iris.info()` and `df_iris.describe()`.
- Visualizing the distribution of features and their relationship with the target variable using Seaborn and Matplotlib.

## Data Preprocessing
- Splitting the dataset into training and testing sets using `train_test_split`.
- Encoding the target variable using Label Encoder and converting it to one-hot encoding.

## Model Architecture
A neural network model is implemented using TensorFlow and Keras with the following layers:
1. Input layer with 128 neurons and ReLU activation.
2. Hidden layer with 64 neurons and ReLU activation.
3. Hidden layer with 32 neurons and ReLU activation.
4. Output layer with 3 neurons and softmax activation.

## Model Training
The model is compiled using the Adam optimizer, categorical crossentropy loss, and accuracy as a metric. It is then trained on the training data for 100 epochs with a batch size of 105.

```python
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_X, train_y_one_hot, epochs=100, batch_size=105)
