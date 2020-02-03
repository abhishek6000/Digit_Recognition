import numpy as np
from forward_propagation import for_prop
from Cost_Neural_Net import cost_function
from Gradient_Descent import grad_Descent
from back_prop import back_prop
from load_mnist import load_data
from Prediction import predict
from Accuracy import calc_accuracy
from Y_lab_test import Y_test
from X_load_test import X_test
def main():
	m = 60000
	print("Loading training data...")
	tem = load_data() # X contains a 2D array 60,000 X 784
	X = np.append(np.ones((m,1)),tem[0], axis=1) # m x (n1+1)
	Y = tem[1] # m x n4
	n1 = 784 
	n2 = 150
	n3 = 150
	n4 = 10
	theta1 = np.random.randn(n1+1,n2) #(n1+1)xn2
	theta2 = np.random.randn(n2+1,n3) #(n2+1)xn3
	theta3 = np.random.randn(n3+1,n4) #(n3+1)xn4
	iti = 800
	alpha = 0.6
	lambu = 100
	theta = [theta1,theta2,theta3]
	print("Learning...")
	for i in range(1,iti,1):
		a = for_prop( theta, X) # a is a 3D matrix
		p = a[3]
		J = cost_function(p,Y,lambu,theta)
		print(J,"\n")
		dW = back_prop(a, theta, Y) #dW is a 3D matrix
		theta = grad_Descent(theta, alpha, dW)
	print("Loading test data...")	
	x_Test = X_test()
	m_test = 10000
	x_test = np.append(np.ones((m_test,1)),x_Test, axis=1)
	y_test = Y_test()
	a = for_prop(theta, x_test) # a is a 3D matrix
	pr = a[3]
	ans = predict(pr)
	accuracy = calc_accuracy(ans, y_test)
	print("Learning Accuracy:  ", accuracy)
	    
main()
