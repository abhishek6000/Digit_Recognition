import numpy as np
def calc_accuracy(p, y):
	truthVal = np.equal(p,y)
	truthVal = truthVal.astype(int)
	m_test = y.shape
	acc = (np.sum(truthVal)/m_test)*100
	return acc

