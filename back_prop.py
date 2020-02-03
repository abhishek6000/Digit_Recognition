import numpy as np
def back_prop(a,theta,y):
    last_layer = a[3] #m x n4
    a3 = a[2] #m x (n3+1)
    a2 = a[1] #m x (n2+1)
    a1 = a[0] #m x (n1+1)
    m=60000
    theta3 = theta[2] #(n3+1) x n4
    theta2 = theta[1] #(n2+1) x n3
    theta1 = theta[0] #(n1+1) x n2	
    del4 = last_layer - y #m x n4
    p3 = np.dot(theta3, np.transpose(del4)) #(n3+1) x m
    gz3 = np.multiply(a3,(1-a3)) #m x (n3+1)
    del3 = np.multiply(np.transpose(p3), gz3) #m x (n3+1)
    d3 = del3[:,1:] #m x n3
    p2 = np.dot(theta2, np.transpose(d3)) #(n2+1) x m
    gz2 = np.multiply(a2,(1-a2)) #m x (n2+1)
    del2 = np.multiply(np.transpose(p2), gz2) #m x (n2+1)
    d2 = del2[:,1:] #mxn2
    dw1 = (np.dot(np.transpose(a1),d2))/m #(n1+1) x n2
    dw2 = (np.dot(np.transpose(a2),d3))/m #(n2+1) x n3
    dw3 = (np.dot(np.transpose(a3),del4))/m #(n3+1) x n4
    dw = [dw1,dw2,dw3] #3D matrix
    #print(dw1[0:6,0:6],"\n")
    #print(dw2[0:6,0:6],"\n")
    #print(dw3[0:6,0:6],"\n")
    return dw


