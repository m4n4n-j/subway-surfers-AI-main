# Implementing Experience Replay using ReplayMemory class.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque

# Saving the tuple of previous state, action, reward and next state, i.e Agent's experiences and storing them in a buffer.
class ReplayMemory:

    # Including N-steps to take into account that our model will be trained on rewards from N steps.
    def __init__(self, n_steps, capacity = 1000):
        self.buffer = deque() #Initializing empty buffer
        self.capacity = capacity #Capacity of our memory
        self.n_steps_iter = iter(n_steps) #Calling n_steps iter function
        self.n_steps = n_steps #Object of n_steps

    # Run the agent for 'n' steps, collect and save the experience in the buffer.
    def run_steps(self, samples): 
        while samples > 0:
            samples -= 1
            entry = next(self.n_steps_iter) #Run game and fill buffer
            self.buffer.append(entry) 
        while len(self.buffer) > self.capacity: #Remove older memory
            self.buffer.popleft()

    # Used to get a batch of 'batch_size' random experiences from the current buffer.
    def sample_batch(self, batch_size): #Random batch generator
        vals = list(self.buffer)
        np.random.shuffle(vals)
        offset = 0
        while (offset+1)*batch_size <= len(self.buffer):
            yield vals[offset*batch_size:(offset+1)*batch_size]
            offset += 1


