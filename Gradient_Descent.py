import numpy as np
def grad_Descent(theta, alpha, dw): #theta and dw are 3D matrices
	theta1 = np.asarray(theta[0])
	theta2 = np.asarray(theta[1])
	theta3 = np.asarray(theta[2])
	dw1 = np.asarray(dw[0])
	dw2 = np.asarray(dw[1])
	dw3 = np.asarray(dw[2])
	theta1 = theta1 - (alpha*dw1)
	theta2 = theta2 - (alpha*dw2)
	theta3 = theta3 - (alpha*dw3)
	theta = [theta1, theta2, theta3]
	return theta