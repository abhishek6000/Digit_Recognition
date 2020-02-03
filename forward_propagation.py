import numpy as np
from Sigmoid import sigmoid 
def for_prop(theta, X): # In a, one row represents one training example, all rows are for input layer only.
# 2nd row is the input layer of 2nd data. Dim = m x (size^2)
# One particular theta 2D array represent propagation from one layer to the other
    theta1 = np.asarray(theta[0])
    theta2 = np.asarray(theta[1])
    theta3 = np.asarray(theta[2])
    (m,n) = X.shape
    a1 = np.asarray(X)
    z2 = np.dot(a1,theta1) #for 2nd hidden layer ALL TRAINING EXAMPLES, m x n2
    z2 = np.asarray(z2)
    a2 = sigmoid(z2)
    a2 = np.append(np.ones((m,1)),a2, axis=1) # m x (n2+1)
    
    z3 = np.dot(a2,theta2)  #for 3rd layer ALL TRAINING EXAMPLES, m x n3
    z3 = np.asarray(z3)
    a3 = sigmoid(z3)
    a3 = np.append(np.ones((m,1)),a3, axis=1) # m x (n3+1)

    z4 = np.dot(a3,theta3) #for 4th layer OUTPUT ALL TRAINING EXAMPLES, m x n4
    z4 = np.asarray(z4)
    a4 = sigmoid(z4)
    
    a = [a1,a2,a3,a4]
    return a    