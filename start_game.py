# Identify the top right and bottom left corners of the nox emulator 
# using predefined images, and return the coordinates.

# Importing the libraries
import pyautogui

# Function identifies the top left and bottom right coordinates of the nox emulator
# Then returns the coordinates.
def begin():
	print("Beginning")
	loc = {}
	loc1 = None
	loc2 = None
	while(loc1 == None):
		loc1 = pyautogui.locateOnScreen('images\start_t.png')
	while(loc2 == None):
		loc2 = pyautogui.locateOnScreen('images\start_b.png')
	loc["top"] = loc1.top + loc1.height
	loc["left"] = loc1.left
	loc["width"] = loc2.left - loc["left"]
	loc["height"] = loc2.top + loc2.height - loc["top"]
	return loc