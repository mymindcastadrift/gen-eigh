import numpy as np
from scipy import linalg

import matrixgenerator as matgen
from choleskywilkinson import cholesky_wilkinson
from fixheiberger import fix_heiberger



# Error functions =====================================

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

# Basic Correctness Tests ==============================================

def run_test(A,M, r):

	[fh_val, fh_vect] = fix_heiberger(A,M,r)
	[cw_val, cw_vect] = cholesky_wilkinson(A,M)
	print "Fix-Heiberger: ", fh_val, " with error: ", average_error(A,M, fh_val, fh_vect)
	print "Cholesky-Wilkinson: ", cw_val, " with error: ", average_error(A,M, cw_val, cw_vect)

def test_correct_1(n):

	print "\nTesting with dimensions:", n

	z = range(1,n+1)
	A = matgen.rand_by_eigenval(n, z[::-1])
	M = matgen.diag([1]*n)
	
	run_test(A,M,0.01)

def test_correct_2(n):

	print "\nTesting with dimensions:", n

	z = range(1,n+1)
	A = matgen.rand_by_eigenval(n, z[::-1])
	M = matgen.rand_by_eigenval(n, matgen.rand_eigenval(n, 1, 10))

	run_test(A,M,0.01)

def test_correct_3(a,d,e):

	print "\nTesting with alpha, delta, epsilon:", a, d, e

	A = np.matrix([[1, a, 0, d],[a, 2, 0, 0], [0,0,3,0],[d,0,0,e]])
	M = matgen.diag([1,1,e,e])

	run_test(A,M,0.01)

def test_correct_4(d):

	print "\n Testing with delta ", d

	A = matgen.diag([6,5,4,3,2,1,0,0])
	A[0,6] = A[1,7] = A[6,0] = A[7,1] = 1
	M = matgen.diag([1,1,1,1,d,d,d,d])

	run_test(A,M,0.00001)

# Unit testing module =====================================================

if __name__ == "__main__":

	print "\nTest 1: Identity M"
	for i in [5,100]:
		test_correct_1(i)

	print "\nTest 2: Non-singular M"
	for i in [5,100]:
		test_correct_2(i)

	print "\nTest 3: Pg 86 test - Limiting values of epsilon and delta"
	for i in range(50,100):
		test_correct_3(0.01, 0.005, 10**(-i))
	for i in range(50,100):
		test_correct_3(0.01, 10**(-i), 0.005)
	# Note how the latter becomes a pathological input for Fix-Heiberger due to the lack of condition (2.14)

	print "\nTest 4: Pg 87 test - Limiting values of delta"
	for i in range(50, 100):
		test_correct_4(10**(-i))

	'''print "\n Test: Higher Dimensional Performance"
	[A, M] = matgen.rand_matrix_pair(1000)
	[eigenval, eigenvect] = fix_heiberger(A,M, 0.1)
	[test_val, test_vect] = cholesky_wilkinson(A,M)
	print "Fix-Heiberger: ", average_error(A,M, eigenval, eigenvect)
	print "Cholesky-Wilkinson: ", average_error(A,M, test_val, test_vect)

	fp = open("testresult.csv", 'w')
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

