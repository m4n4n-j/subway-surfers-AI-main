# Importing the libraries
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import numpy as np
from collections import namedtuple, deque

#This step/experience includes the state of game, action performed, the reward for the state
# whether the game had ended after this state and the LSTM values
Step = namedtuple('Step', ['state', 'action', 'reward', 'done', 'lstm']) #This is a storage of one step that our agent performs


class NStepProgress:
    
    def __init__(self, env, ai, n_step):
        self.ai = ai #ai object
        self.rewards = []
        self.env = env #Importing our manual subway surfers enviornment
        self.n_step = n_step #Number of steps to look forward

    def __iter__(self): #Function to play game and collect/return samples
        state = self.env.reset() #Resetting the game by clicking the green play button
        history = deque() 
        reward = 0.0 #Initial reward = 0
        is_done = True
        end_buffer = [] #To remove unwanted images that were added because of the late detection of end of game

        while True:
            if is_done:
                cx = Variable(torch.zeros(1,256))
                hx = Variable(torch.zeros(1,256))
            else:
                cx = Variable(cx.data)
                hx = Variable(hx.data)

            action, (hx, cx) = self.ai(Variable(torch.from_numpy(np.array([state], dtype = np.float32))), (hx, cx)) #Calculating action for a state/image
            end_buffer.append((state, action))
            
            while len(end_buffer) > 3: #By observation at max 3 extra steps played by the agent after dying so a buffer of size 3
                del end_buffer[0]

            # Printing action output from softmax.
            t = action[0][0]

            if(t == 1):     #left
                print("left")
            elif (t == 2):  #right
                print("right")
            elif (t == 3):  #roll
                print("jump")
            elif (t == 4):  #jump
                print("roll")
            elif (t == 0):  #no op
                print("do nothing")

            # Taking Action
            next_state, r, is_done, _ = self.env.step(action) #Performing action and returning next state

            # If game over, 
            if(is_done):
                print("\nGame Ended\n")
                if len(end_buffer)>=3:
                    state, action = end_buffer[-3]
                    history.pop() #Removing unwanted experience that includes images such as when cop was holding the runner
                r=-10
            reward += r
            history.append(Step(state = state, action = action, reward = r, done = is_done, lstm = (hx, cx))) #Adding this experience to history
            
            # Returning the experiences generated to the replay memory
            while len(history) > self.n_step + 1:
                history.popleft()
            if len(history) == self.n_step + 1:
                yield tuple(history)
            state = next_state
            # If game is over
            if is_done:
                if len(history) > self.n_step + 1:
                    history.popleft()
                while len(history) >= 1:
                    yield tuple(history)
                    history.popleft()
                #Resetting the variables
                self.rewards.append(reward)
                reward = 0.0
                state = self.env.reset()
                end_buffer=[]
                history.clear()
    
    def rewards_steps(self): #Function to store the rewards whenever the game ends/for one instance of game
        rewards_steps = self.rewards
        self.rewards = []
        return rewards_steps