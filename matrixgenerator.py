import numpy as np
import numpy.matlib
from scipy import linalg

def rand_symm(n, range_a = 100):
	A = (2*np.matlib.rand(n,n) - 1)*range_a
	A = (A + A.getH())/2
	return A

def rand_semidef_symm(n, range_m = 10):
	M = (2*np.matlib.rand(n,n) - 1)*range_m
	M = M.getH() * M
	return M

def rand_semidef_diag(n, range_d = 1000, return_list = False):
	v = np.random.rand(n) * range_d
	if return_list:
		return np.diag(v), v
	else:
		return np.diag(v)

def rand_unitary(n):
	A = 2*np.matlib.rand(n,n) - 1
	[Q,R] = linalg.qr(A, overwrite_a = true)
	return Q

def rand_matrix_pair(n, range_m = 10, range_a = 100):
	return rand_symm_matrix(n, range_a), rand_semidefinite_symm_matrix(n, range_m)
