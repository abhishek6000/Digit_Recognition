import numpy as np
import gzip
import struct
def Y_test():
    unload = gzip.open("t10k-labels-idx1-ubyte.gz","rb")
    print(struct.unpack('>I',unload.read(4))[0])
    print(struct.unpack('>I',unload.read(4))[0])
    opu=[]
    for i in range(0,10000):
        opu.append(struct.unpack('>B',unload.read(1))[0])
    y=np.asarray(opu)
    y = np.transpose(y)
    return y






