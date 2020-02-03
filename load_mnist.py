#Your image is considered to be in B/W and preprocessed to 250X250

import struct 
import numpy as np
import gzip
import matplotlib.pyplot as plt
def load_data(): #The value of p is the pth digit of the dataset
    img = gzip.open("train-images-idx3-ubyte.gz", "rb")
    pic = gzip.open("train-labels-idx1-ubyte.gz","rb")
    print(struct.unpack('>I',img.read(4))[0])
    print(struct.unpack('>I',img.read(4))[0])
    print(struct.unpack('>I',img.read(4))[0])
    print(struct.unpack('>I',img.read(4))[0])

    print(struct.unpack('>I',pic.read(4))[0])
    print(struct.unpack('>I',pic.read(4))[0])
    y=np.zeros((60000,10))
    data=[]
    c=[]
    temp = []
    for x in range(0,60000): # 
    
        a = []

        temp = struct.unpack('>B',pic.read(1))[0] # Getting the original values of Y
        for i in range(0,28*28):
            
            a.append(struct.unpack('>B',img.read(1))[0])
        
        data.append(a)
        
        for j in range(0,10): #For loading the labels
            if(j==temp):
                y[x,j] = 1
            
    y = np.asarray(y)
    data = np.asarray(data)
    ultimate = (data,y)        
    return ultimate

        
    #return(ultimate) #data is a 3D arraycl

