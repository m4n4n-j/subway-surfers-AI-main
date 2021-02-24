# Importing the libraries
import time
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib.pyplot as pyplot
import pandas as pd

# Importing the other Python files
from env import env
import n_step
import replay_memory
import neural_net
from eligibility_trace import eligibility_trace
import moving_avg

#If OMP Error comes then paste following commands to python console
'''
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
'''

# Getting the Subway Surfers environment
senv = env()
number_actions = senv.action_space

# Building an AI
cnn = neural_net.CNN(number_actions)
softmax_body = neural_net.SoftmaxBody(T = 10)
ai = neural_net.AI(body = softmax_body, brain = cnn)

# Setting up Experience Replay and n_step progress
n_steps = n_step.NStepProgress(ai = ai, env = senv, n_step = 7)
memory = replay_memory.ReplayMemory(n_steps = n_steps, capacity = 5000)

ma = moving_avg.MA(500) #Moving average used to grade our model

# Functions to save and load the checkpoints created while training.
def load():
    if os.path.isfile('old_brain.pth'):
        print("=> loading checkpoint... ")
        checkpoint = torch.load('old_brain.pth')
        cnn.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("done !")
    else:
        print("no checkpoint found...")
        
def save():
        torch.save({'state_dict': cnn.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                   }, 'old_brain.pth')


# Training the AI
nb_epochs = 200 #Modify this to get better results (We were able to train only for 20 epochs at once)
optimizer = optim.Adam(cnn.parameters(), lr = 0.005) #Using Adam optimizer
loss = nn.MSELoss() #Using Mean Squared Error loss

#Uncomment if you have old_brain to use
#load() 

#Training begins here!
for epoch in range(1, nb_epochs + 1):
    print("Playing game for Epoch : %s" %str(epoch))
    print("Printing actions")
    memory.run_steps(128) #Calling n_steps 128 times and filling the buffer
    print("Entering Epoch :")
    for batch in memory.sample_batch(64): #Randomly choosing 64 samples
        inputs, targets = eligibility_trace(batch, cnn) # Calculate Target Qvalues for comparision and evaluating our model.
        inputs, targets = Variable(inputs), Variable(targets)
        predictions, hidden = cnn(inputs, None)
        loss_error = loss(predictions, targets) #Calculating loss
        optimizer.zero_grad() #Setting gradients to zero
        loss_error.backward() #Doing back propagation
        optimizer.step() #Updating weights

    #Evaluating our model on games played in this epoch
    rewards_steps = n_steps.rewards_steps()
    ma.add(rewards_steps) 
    avg_reward = ma.average() #Calculating average of rewards
    print("Epoch: %s, Average Reward: %s" % (str(epoch), str(avg_reward))) #Output for each epoch
    save() #Saving current model
    #Note: these rewards are not the scores displayed at the end of games. They are the number of steps taken*2 and still the agent is alive
    if avg_reward >= 20: #Checking for some milestones
        print("20 reached")
        save()
    if avg_reward >= 50: #Checking for some milestones
        print("50 reached")
        save()                
    if avg_reward >= 100: #This score is really great
        print("Congratulations!")
        save()                
        break
