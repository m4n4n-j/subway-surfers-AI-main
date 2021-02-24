# The definitons and declarations of the neural network structure and the softmax function.

# Importing the libraries
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as pyplot
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Making the brain
class CNN(nn.Module):
    # Defining the structure of the neural network.
    # 3 convolutional layers -> 1 Lstm layer -> 2 linear layers.
    def __init__(self, number_actions):
        super(CNN, self).__init__()
        self.convolution1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 5) #32 filters and kernel size of 5X5
        self.convolution2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3) #32 filters and kernel size of 3X3
        self.convolution3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 2) #32 filters and kernel size of 2X2
        self.out_neurons = self.count_neurons((1, 128, 128)) #Calculating number of output neurons from convolution layers
        self.lstm = nn.LSTMCell(self.count_neurons((1, 128, 128)), 256) #LSTM layer (input = 10816 & output = 256)
        self.fc1 = nn.Linear(in_features = 256, out_features = 40) #Fully connected layer (input = 256 & output = 40)
        self.fc2 = nn.Linear(in_features = 40, out_features = number_actions) #Fully connected layer (input = 40 & output = 5)

    # Function for forward propagation of data in the NN.
    def forward(self, x, hidden=None):
        x = F.relu(F.max_pool2d(self.convolution1(x), 3, 2)) #Applying maxpool and then Relu
        x = F.relu(F.max_pool2d(self.convolution2(x), 3, 2)) #Applying maxpool and then Relu
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2)) #Applying maxpool and then Relu
        x = x.view(-1, self.out_neurons) #Flattening the layer
        hx, cx = self.lstm(x, hidden) 
        x = hx #One output of LSTM is same as x
        x = F.relu(self.fc1(x))
        x = self.fc2(x) #Final output (a vector of size 5)
        return x, (hx, cx)

    # Function to count the number of neurons in LSTM layer. 
    def count_neurons(self, image_dim):
        x = Variable(torch.rand(1, *image_dim))
        x = F.relu(F.max_pool2d(self.convolution1(x), 3, 2)) #Applying maxpool of kernel size 3X3 and stride of 2
        x = F.relu(F.max_pool2d(self.convolution2(x), 3, 2)) #Applying maxpool of kernel size 3X3 and stride of 2
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2)) #Applying maxpool of kernel size 3X3 and stride of 2
        return x.data.view(1, -1).size(1)

# Declaration of AI model, which does forward propagation using NN and softmax.
class AI:

    def __init__(self, brain, body):
        self.brain = brain
        self.body = body
# Returns the output of Softmax multinomial function and LSTM layer
    def __call__(self, inputs, hidden):
        output, (hx, cx) = self.brain(inputs,hidden)
        actions = self.body(output)
        return actions.data.cpu().numpy(), (hx, cx)

# Body of the softmax function.
class SoftmaxBody(nn.Module):
    
    def __init__(self, T):
        super(SoftmaxBody, self).__init__()
        self.T = T #Used for random exploration (the higher the T the lower the exploration)

    def forward(self, outputs):
        probs = F.softmax(outputs * self.T,dim = 0) 
        actions = probs.multinomial(num_samples=1)  # Final action to perform
        return actions