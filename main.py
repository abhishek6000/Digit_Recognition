import random
from forward_propagation import for_prop
from Cost_Neural_Net import cost_function
from Gradient_Descent import grad_Descent
from back_prop import back_prop
from load_mnist import load_data
from Prediction import predict
from Accuracy import calc_accuracy
def main():
	m = 60000
	print("Loading training data...")
	tem = load_data() # X contains a 2D array 60,000 X 784
	X = np.append(np.ones((m,1)),tem[0], axis=1) # m x (n1+1)
	Y = tem[1] # m x n4
	n1 = 784
	n2 = 150
	n3 = 100
	n4 = 10
	theta1 = np.random.randn(n1+1,n2) #(n1+1)xn2
	theta2 = np.random.randn(n2+1,n3) #(n2+1)xn3
	theta3 = np.random.randn(n3+1,n4) #(n3+1)xn4
	iti = 100
	alpha = 0.1
	lambu = 100
	theta = (theta1,theta2,theta3)
	print("Learning...")
	for i in range(1,iti,1):
		a = for_prop(theta1 , theta2, theta3, X) # a is a 3D matrix
		p = a[3]
		J = cost_function(p,Y,lambu,theta)
		dW = back_prop(a, theta, Y) #dW is a 3D matrix
		theta = grad_Descent(theta, alpha, dW)
	print("Loading test data...")	
	X_test =
	y_test =
	a = for_prop(theta1 , theta2, theta3, X_test) # a is a 3D matrix
	p = a[3]
	ans = predict(p)
	accuracy = calc_accuracy(ans, y_test)
	print("Learning Accuracy:  ", accuracy)