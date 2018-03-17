def loss_function_x(x):
	if x<0:
		return 0
	else:
		return x
		
def dloss_function_dx_x(x):
	if x<0:
		return 0
	else:
		return 1
