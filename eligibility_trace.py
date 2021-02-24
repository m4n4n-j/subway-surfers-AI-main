import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# Implementing Eligibility Trace
def eligibility_trace(batch, cnn):
    targets = [] #Target for evaluation of our model
    inputs = []
    gamma = 0.99 #Gamma to reduce effect of older rewards
    for series in batch:
        input = Variable(torch.from_numpy(np.array([series[0].state, series[-1].state], dtype = np.float32))) #Extracting input
        output, hidden = cnn(input) #Forward propagation
        cumul_reward = 0.0 if series[-1].done else output[1].data.max() # Defining cumulative reward
        for step in reversed(series[:-1]): #This is a named tuple that we defined in n_step.py
            cumul_reward = step.reward + gamma * cumul_reward #Reducing effect of older rewards
        target = output[0].data
        target[series[0].action] = cumul_reward
        state = series[0].state
        targets.append(target)
        inputs.append(state)
    return torch.from_numpy(np.array(inputs, dtype = np.float32)), torch.stack(targets)