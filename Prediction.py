import numpy as np
def predict(ann):
	prec = np.argmax(ann,axis=1)
	prec = np.transpose(prec)
	return prec
