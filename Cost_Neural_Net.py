import numpy as np
def cost_function(p,y,lab, theta): #theta is a 3D matrix, Lab -> lambda(regularization term), p -> output of the O/P layer (each element is b/w 0-1) 
    (m,n) = y.shape 
    theta1 = theta[0]
    theta2 = theta[1]
    theta3 = theta[2]
    theta_sq_sum = np.sum(theta1[:,1:]**2) + np.sum(theta2[:,1:]**2) + np.sum(theta3[:,1:]**2)

    costi = np.multiply(y, np.log(p)) + np.multiply((1-y), np.log((1-p)))
    J =  -(np.sum(costi))/m #(lab*0.5*theta_sq_sum)/m
    return J