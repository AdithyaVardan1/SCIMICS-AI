# Chatbot Project

This project implements a simple chatbot using a neural network for intent recognition. The chatbot is designed to respond to user input based on predefined intents. The neural network is trained to classify user input into specific intent categories.

## Neural Network Architecture

The neural network used in this project is a basic feedforward neural network with the following architecture:

- Input Layer: The size is determined by the number of features in the input data.
- Hidden Layers: Two hidden layers with ReLU activation functions.
- Output Layer: The size is determined by the number of intent categories.

The architecture is defined in the `ChatModel` class within the `model.py` file.

## Data Preprocessing

The training data for the neural network is loaded from a JSON file (`intents.json`). The input patterns are tokenized and converted into a bag-of-words representation. The words are stemmed and preprocessed. The intents and patterns are used to create training data for the neural network.

## Training

The model is trained using the PyTorch framework. The hyperparameters used for training are as follows:

- Number of Epochs: 1000
- Batch Size: 8
- Learning Rate: 0.001

The loss function used is CrossEntropyLoss, and the Adam optimizer is employed for gradient descent.

## Model Usage

The trained model can be loaded using the `ChatModel` class and applied to classify user input into specific intents. The chat loop is implemented in the `main.py` file.

## Intent Examples

The predefined intents in the `intents.json` file include:

- Greeting
- Goodbye
- Thanks
- Machine Learning Engineer
- Full Stack Developer
- Cyber Security Professional
- IoT Engineer
- Data Analyst
- Game Developer
- Blockchain Engineer
- App Developer

## Dependencies

Ensure you have the following dependencies installed:

- Python
- PyTorch
- NumPy

