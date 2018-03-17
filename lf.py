import numpy as np

def loss_function_x(x):
	x*(np.sign(x)+1)/2
		
def dloss_function_dx_x(x):
	(np.sign(x)+1)/2
