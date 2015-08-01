import numpy as np
import numpy.matlib
from scipy import linalg

import matrixgenerator as matgen
from choleskywilkinson import cholesky_wilkinson
from fixheiberger import fix_heiberger


def column_vect_norms(X):
	norm =[]
	for i in range(X.shape[1]):
		x = linalg.norm(X[:,i])
		norm.append(x)
	return norm

def average_error(A, M, eigenval, eigenvect):
	err_matrix = A.dot(eigenvect) - np.multiply(eigenval, M.dot(eigenvect))
	norm = column_vect_norms(err_matrix)
	return sum(norm)/len(norm)


if __name__ == "__main__":

	[A, M] = matgen.rand_symm_matrices(1000)
	[eigenval, eigenvect] = fix_heiberger(A,M, 0.1)
	[test_val, test_vect] = cholesky_wilkinson(A,M)
	print "Fix-Heiberger: ", average_error(A,M, eigenval, eigenvect)
	print "Cholesky-Wilkinson: ", average_error(A,M, test_val, test_vect)

	'''fp = open("testresult.csv", 'w')
	print "Beginning Testing ... "
	for n in range(1,100):
		fp.write("{}".format(n*10))
		for r in range (1,5):
			[A, M] = rand_symm_matrices(10*n)
			[eigenval, eigenvect] = fix_heiberger(A,M, 0.00001)
			[test_val, test_vect] = cholesky_wilkinson(A,M)
			err_1 = average_error(A, M , eigenval, eigenvect)
			err_2 = average_error(A, M, test_val, test_vect)
			fp.write(",{},{}".format(err_1, err_2))
		fp.write("\n")
	fp.close()''' 

