# Return the moving average of last 100 observations.
# To gauge the performance of our model in the last 100 steps at the end of every epoch.
# 100 is a random number , can be anything.

import numpy as np

# Making the moving average on 100 steps
class MA:
    def __init__(self, size):
        self.size = size #Size of moving average
        self.list_of_rewards = []
        

    # Function to Calculate average and return 
    def average(self):
        return np.mean(self.list_of_rewards)

    # Function to Append rewards to the list of rewards.
    def add(self, rewards):
        if isinstance(rewards, list):
            self.list_of_rewards = self.list_of_rewards + rewards
        else:
            self.list_of_rewards.append(rewards)
        # Removing elements from beginning if no of elements > size
        while len(self.list_of_rewards) > self.size:
            del self.list_of_rewards[0]