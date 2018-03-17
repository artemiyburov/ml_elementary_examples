import numpy as np

def loss_function(x):
	if x<0:
		return 0
	else:
		return x
		
def dloss_function_dx(x):
	if x<0:
		return 0
	else:
		return 1
