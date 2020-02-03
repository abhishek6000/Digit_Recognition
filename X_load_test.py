import numpy as np
import gzip
import struct
def X_test():
	pic = gzip.open("t10k-images-idx3-ubyte.gz","rb")

	print(struct.unpack('>I',pic.read(4))[0])
	print(struct.unpack('>I',pic.read(4))[0])
	print(struct.unpack('>I',pic.read(4))[0])
	print(struct.unpack('>I',pic.read(4))[0])
	ar=[] 
	for i in range(0,10000):
		temp=[]  

		for j in range(0,28*28):
			k = struct.unpack('>B',pic.read(1))[0]
			temp.append(k)
		ar.append(temp)
	ar=np.asarray(ar)
	return ar
