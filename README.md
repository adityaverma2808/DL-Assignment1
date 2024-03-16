---
# Feed Forward Neural Network

## Deep Learning (CS6910) Assignment-1

This repository contains a Python script that uses NumPy exclusively to train a feed forward neural network. The neural network is an excellent choice for classifying datasets such as MNIST or Fashion MNIST because of its adaptable design that allows for a wide range of setups. The integration of activation functions, loss functions, and extra parameters as needed is made easy by the smooth customization process.

# train.py file
The primary file (a Python script) that has to be uploaded is this one. The assignment description's stated commandline arguments can be entered using it.

When the file is executed without the commandline arguments, it uses default parameters, which are optimized hyperparameter values discovered during tuning.

The file runs for specified values when run with the command-line arguments.

The codes listed below have been commented after they have been run once and their plots reported. If you want to test certain features, you have to uncomment these.

1. Sweep : Lines of the Wandb agent has been commented out; uncomment this line to perform sweeps.
2. Confusion Matrix Plot: To plot the confusion matrix, uncomment lines .
The MNIST dataset configurations consist of setting the dataset value to'mnist' using commandline arguments and uncommenting lines.


## Features

- A configurable neural network design.
- Support for sigmoid, tanh, identity, and ReLU activation functions, among others.
- Two different loss functions: Mean Squared Error and Cross Entropy.
- A range of optimization methods, including Adam, Nadam, RMSprop, Momentum, SGD, and NAG.
- Xavier and Random weight initialization techniques.
- Metric visualization with weights and biases.

# Best Parameters

epochs = 10

batch size = 32

loss function = 'cross_entropy'

optimizer = 'nadam'

learning rate = 1e-3

weight decay constant = 0.0005

weight initialization = 'Xavier'

number of hidden layers = 3

hidden layer size = 128

activation function = 'tanh'


# Best Results Obtained

Train accuracy:  88.25%

Validation accuracy : 87.09%

Test accuracy : 87.67%



## Parameters

- `-d`, `--dataset`: Choose the dataset for training ("mnist" or "fashion_mnist").
- `-e`, `--epochs`: Number of training epochs.
- `-b`, `--batch_size`: Batch size for training.
- `-l`, `--loss`: Loss function for training ("mean_squared_error" or "cross_entropy").
- `-o`, `--optimizer`: Optimization algorithm ("sgd", "momentum", "nag", "rmsprop", "adam", "nadam").
- `-lr`, `--learning_rate`: Learning rate for optimization.
- `-m`: Momentum for Momentum and NAG optimizers.
- `-beta1`, `--beta1`: Beta1 parameter for Adam and Nadam optimizers.
- `-beta2`, `--beta2`: Beta2 parameter for Adam and Nadam optimizers.
- `-w_i`, `--weight_init`: Weight initialization method ("random" or "Xavier").
- `-nhl`, `--num_layers`: Number of hidden layers in the neural network.
- `-sz`, `--hidden_size`: Number of neurons in each hidden layer.
- `-a`, `--activation`: Activation function for hidden layers ("identity", "sigmoid", "tanh", "ReLU").
- `-oa`, `--output_activation`: Activation function for output layer ("softmax").
- `-cl`, `--console_log`: Log training metrics on Console (0: disable, 1: enable).
- `-wl`, `--wandb_log`: Log training metrics on Weights & Biases (0: disable, 1: enable).
- `-cm`, `--confusion_matrix`: Plot and show confusion matrix (0: disable, 1: enable).

# How to Train a Model

To train a model, run:

```bash
python train.py --wandb_entity myname --wandb_project myprojectname