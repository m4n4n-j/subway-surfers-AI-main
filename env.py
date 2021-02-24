# Creating an environment class for starting or restarting the game, by scanning the play.png image on the screen.

# Importing the libraries
import numpy as np
import pyautogui
import time
from pynput import keyboard
import random

#Importing the files
from action import action
from start_game import begin
from preprocess_image import preprocess_image


class env:
    def __init__(self):
        self.action_space = 5
        self.loc = begin()
        pyautogui.click(x=self.loc["left"]+self.loc["width"]/2, y=self.loc["top"]+self.loc["height"]/2, clicks=1, button='left')
        self.act = action(self.loc["left"], self.loc["top"], self.loc["width"], self.loc["height"])     
        
    # To take random action.
    def action_space_sample(self):
        return random.randint(0,4)
    
    #  While play button not found, keep checking every .1 seconds. 
    #  If play button visible, click on it and send the image after waiting # 2.5 seconds. 
    def reset(self):
        while(pyautogui.locateOnScreen('images\play.png', confidence=.7) == None):
            time.sleep(0.1)
        x, y = pyautogui.locateCenterOnScreen('images\play.png', confidence=.7)
        pyautogui.click(x, y)
        time.sleep(2.5)
        state = preprocess_image(pyautogui.screenshot(region =(self.loc["left"], self.loc["top"], self.loc["width"], self.loc["height"])))
        return state

    # After each step check if game over.
    # Wait for .2 seconds after taking action and see if play.png is in the frame.
    # If play.png is not visible, return next_state.
    # if game over, reward = -5 else reward = 1.
    def step(self, action):
        self.act.perform(action)
        Done = True
        if pyautogui.locateOnScreen('images\play.png', confidence=.7) == None:
            Done = False
        next_state = preprocess_image(pyautogui.screenshot(region =(self.loc["left"], self.loc["top"], self.loc["width"], self.loc["height"])))
        reward = 2
        if(Done):
            reward = -10
        return (next_state, reward, Done, {})

