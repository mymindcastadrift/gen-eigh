import numpy as np
import numpy.matlib

def rand_symm_matrices(n, range_m = 10, range_a = 100):
	M = (2*np.matlib.rand(n,n) - 1)*range_m
	M = M.getH() * M
	A = (2*np.matlib.rand(n,n) - 1)*range_a
	A = (A + A.getH())/2
	return (A,M)