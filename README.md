# Body Parts Identifier using CNN

## Overview

This project is an implementation of a Convolutional Neural Network (CNN) for the identification of body parts in X-ray images. It uses the UNIFESP X-Ray Body Part Classification Dataset obtained from Kaggle. The project is hosted on Google Colab and involves data preprocessing, model training, and evaluation.

## Project Structure

`bodyparts.ipynb`: A Jupyter Notebook containing the project code.You can access the notebook and run the code by clicking on the following link:

[Google Colab Notebook](https://colab.research.google.com/drive/1TV-N7uGm36CF5fSf3a7dt8TdRCZmzmqS?usp=sharing)

## Requirements

-  Python
-  Tensorflow
-  Numpy
-  Matplotlib
-  Scikit-Learn
-  Kaggle dataset
-  cv2
-  os

## Setup

### Data Preparation

- Download the [UNIFESP X-Ray Body Part Classification Dataset from Kaggle](https://www.kaggle.com/datasets/felipekitamura/unifesp-xray-bodypart-classification?select=train.csv)
- Upload the dataset to your Google Drive.
- Open the bodyparts.ipynb notebook in Google Colab and ensure that it's connected to your Google Drive.

Run the notebook step by step for data preprocessing, model training, prediction, and evaluation.

## Data Preprocessing

The data preprocessing steps in the notebook include:

- Loading the dataset from Google Drive.
- Splitting the dataset into training, validation, and test sets.
- Preprocessing images (resizing, normalization).
- Model
  
The model used for this project is a CNN with 3 convolutional layers and max-pooling. You can find the architecture details in the notebook.

## Training

The training section includes:

1. Model compilation and configuration.
2. Training the model on the training dataset.
3. Monitoring training progress and visualizing metrics.
4. Prediction
5. 
After model training, predictions are performed on both the validation and test datasets. The predictions are saved as X-ray images with the predicted body part labeled.

## Evaluation

A confusion matrix is created to visualize the model's predictions and assess its performance.

## Author

Yuktha PS
