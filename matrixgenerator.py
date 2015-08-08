import numpy as np
import numpy.matlib
from scipy import linalg

def rand_mat(n,m, range_low = -1000, range_high=1000):
	A = np.matlib.rand(n,m)*(range_high-range_low) + range_low
	return np.matrix(A)

def rand_symm(n, range_a = 100):
	A = (2*np.matlib.rand(n,n) - 1)*range_a
	A = (A + A.getH())/2
	return A

def rand_semidef_symm(n, range_m = 10):
	M = (2*np.matlib.rand(n,n) - 1)*range_m
	M = M.getH() * M
	return M

def rand_by_eigenrange(n, range_low = 0, range_high =1000):
	return rand_by_eigenval(n, rand_eigenval(n, range_low, range_high))

def rand_unitary(n):
	A = 2*np.matlib.rand(n,n) - 1
	[Q,R] = linalg.qr(A, overwrite_a = True)
	return np.matrix(Q)

def rand_pair(n, range_m = 10, range_a = 100):
	return rand_symm(n, range_a), rand_semidef_symm(n, range_m)

def rand_by_eigenval(n, eigenval):
	Q = rand_unitary(n)
	D = np.diag(eigenval)
	return Q.getH() * D * Q

def diag(z):
	return np.matrix(np.diag(z))

def rand_eigenval(n, range_low = 0, range_high = 1000):
	z = np.random.rand(n)
	z = [i *(range_high - range_low) + range_low for i in z]
	return z