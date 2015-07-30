import numpy as np
import numpy.matlib
from scipy import linalg

# Implementation with Triangular_Solve, but no LAPACK wrapped
# Highly unstable for large dim > 1000

def cholesky_wilkinson(A, M): #A and M are both NumPy Matrix Objects
	L =  np.matrix(linalg.cholesky(M))
	S = np.matrix(linalg.solve_triangular(L, A))   # S = LG
	G = (np.matrix(linalg.solve_triangular(L, S.getH()))).getH()
	[eigenval, eigenvect] = linalg.eigh(G)
	eigenvect = np.matrix(linalg.solve_triangular(L, eigenvect, trans = 'C'))
	eigenval = np.matrix(eigenval)
	return (eigenval, eigenvect)



# Key Areas of Improvement:
# 1) Do not store the matrix if it is not necessary to do so, eg. L_h


def rand_symm_matrices(n, range_m = 10, range_a = 100):
	M = (2*np.matlib.rand(n,n) - 1)*range_m
	M = M.getH() * M
	A = (2*np.matlib.rand(n,n) - 1)*range_a
	A = (A + A.getH())/2
	return (A,M)

if __name__ == "__main__":

	print "Beginning Testing ... "
	[A, M] = rand_symm_matrices(4)
	print "A: ", A
	print "M: ", M
	[test_val, test_vect] = cholesky_wilkinson(A,M)

	print "Cholesky-Wilkinson: ", test_val, "with error: ", linalg.norm(A.dot(test_vect) - np.multiply(test_val, M.dot(test_vect)))