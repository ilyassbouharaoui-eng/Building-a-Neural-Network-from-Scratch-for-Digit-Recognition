# DigitNet - Neural Network from Scratch for Image Classification

A simple neural network implemented **from scratch using NumPy** to classify handwritten digits.  
This project aims to understand the fundamentals of deep learning by implementing the complete training process without using deep learning frameworks such as TensorFlow or PyTorch.

## Project Overview

The goal of this project is to build a neural network capable of recognizing handwritten digits (0-9) from small grayscale images.

The model is trained on the `digits` dataset provided by Scikit-learn, where each image is:

- Size: 8 × 8 pixels
- Input features: 64 pixels
- Classes: 10 digits (0 to 9)

The neural network learns the relationship between pixel values and digit labels using forward propagation, backpropagation, and gradient descent.

## Neural Network Architecture

The implemented network contains:
```text
+-----------------------+
|      Input Layer      |
|    64 pixels (8x8)    |
+-----------------------+
            |
            v
+-----------------------+
|    Hidden Layer       |
|  16 neurons + ReLU    |
+-----------------------+
            |
            v
+-----------------------+
|     Output Layer      |
| 10 neurons + Softmax  |
+-----------------------+
            |
            v
+-----------------------+
|  Predicted digit      |
|       0 - 9           |
+-----------------------+
```
