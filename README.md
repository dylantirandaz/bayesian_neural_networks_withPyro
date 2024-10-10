# MyFirstBNN - Bayesian Neural Network

## Overview
This repository contains an implementation of a Bayesian Neural Network (BNN) using the Pyro library, which is built on top of PyTorch. The BNN is designed to model uncertainty in predictions and learn from input data while incorporating prior beliefs about the model parameters.

## Code Description

### Imports
The code imports necessary libraries:
- **Pyro** for probabilistic programming.
- **PyTorch** for neural network components.
- **nn** module for neural network layers.

### Class Definition
The `MyFirstBNN` class inherits from `PyroModule`, indicating that it is a Pyro-compatible neural network module.

### Initialization (`__init__` method)
The constructor initializes the neural network with:
- **Input dimension** (`in_dim`), **output dimension** (`out_dim`), **hidden dimension** (`hid_dim`), and **prior scale** (`prior_scale`).
- Two linear layers (`layer1` and `layer2`) are created using `PyroModule`, allowing for probabilistic sampling of weights and biases.
- The weights and biases of both layers are initialized as samples from normal distributions, defined as priors.

### Activation Function
An activation function (Tanh) is applied to the output of the first layer. Note: There is a code issue where `self.activation` should be assigned with `=` instead of `==`.

### Forward Method
The `forward` method defines how the input data flows through the network:
1. The input `x` is reshaped and passed through the first layer with the activation function applied.
2. The output of the first layer is passed through the second layer to produce the mean (`mu`).
3. A sample for the standard deviation (`sigma`) is drawn from a Gamma distribution.
4. The observed data is modeled using a Normal distribution, where the mean is `mu` and the standard deviation is `sigma`.

### Probabilistic Modeling
The code uses Pyro's `plate` construct to indicate that the observations are independent and identically distributed (i.i.d.) across the data points.

## Summary
This implementation of a Bayesian Neural Network allows for uncertainty quantification in predictions, making it suitable for applications where understanding the confidence of predictions is crucial. The network learns from input data while incorporating prior beliefs about the weights and biases through probabilistic sampling.
