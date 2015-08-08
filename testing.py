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
	[def_val, def_vect] = linalg.eigh(A,M)
	print "Fix-Heiberger:  with error: ", average_error(A,M, fh_val, fh_vect)#, "\n", fh_val
	print "Cholesky-Wilkinson:  with error: ", average_error(A,M, cw_val, cw_vect)#, "\n", cw_val
	print "Scipy: with error: ", average_error(A,M, def_val, def_vect), "\n", #def_val, "\n"

def test_correct_1(n):

	print "Testing with dimensions:", n

	z = range(1,n+1)
	A = matgen.rand_by_eigenval(n, z[::-1])
	M = matgen.diag([1]*n)
	
	run_test(A,M,0.01)

def test_correct_2(n):

	print "Testing with dimensions:", n

	A = matgen.rand_symm(n)
	M = matgen.rand_by_eigenval(n, matgen.rand_eigenval(n, 1, 10))

	run_test(A,M,0.01)

def test_correct_3(a,d,e):

	print "Testing with alpha, delta, epsilon:", a, d, e

	A = np.matrix([[1, a, 0, d],[a, 2, 0, 0], [0,0,3,0],[d,0,0,e]])
	M = matgen.diag([1,1,e,e])

	run_test(A,M,0.01)

def test_correct_4(d):

	print "Testing with delta ", d

	A = matgen.diag([6,5,4,3,2,1,0,0])
	A[0,6] = A[1,7] = A[6,0] = A[7,1] = 1
	M = matgen.diag([1,1,1,1,d,d,d,d])

	run_test(A,M,0.000011)

def test_correct_5(n,w):

	print "Testing with negative eigenvalues in A_22"

	A_11 = matgen.rand_symm(n)
	A_22 = matgen.rand_by_eigenval(w, matgen.rand_eigenval(w,-1000,1000))
	A_13 = matgen.rand_mat(n,w)
	A = linalg.block_diag(A_11, A_22)
	A[0:n, n:n+w] = A_13
	A[n:n+w, 0:n] = A_13.getH()

	M =linalg.block_diag( matgen.diag(matgen.rand_eigenval(n,100,1000)), matgen.diag([10**-60]*w))

	run_test(A,M,0.01)

# Unit testing module =====================================================

if __name__ == "__main__":

	print "\nTest 1: Identity M"
	for i in [5,100]:
		test_correct_1(i)

	print "\nTest 2: Non-singular M"
	for i in [5,100]:
		test_correct_2(i)

	print "\nTest 3: Pg 86 test - Limiting values of epsilon"
	for i in range(50,60):
		test_correct_3(0.01, 0.005, 10**(-i))

	print "\nTest 3: Pg 86 test - Limiting values of delta"
	for i in range(50,60):
		test_correct_3(0.01, 10**(-i), 0.005)
	# Note how the latter becomes a pathological input for Fix-Heiberger due to the lack of condition (2.14)

	print "\nTest 4: Pg 87 test - Limiting values of delta"
	for i in range(5, 10):
		test_correct_4(10**(-i))

	print "\nTest 5: A_22 with negative values"
	for i in range(1, 5):
		test_correct_5(10,2)
	# Note that higher error for "less singular matrices" is expected.

	print "\n Test: Higher Dimensional Performance"
	[A, M] = matgen.rand_pair(1000)
	run_test(A,M, 0.0001)

	fp = open("testresult.csv", 'w')
	print "Beginning Testing ... "
	for n in range(1,100):
		fp.write("{}".format(n*10))
		for r in range (1,5):
			[A, M] = matgen.rand_pair(10*n)
			[eigenval, eigenvect] = fix_heiberger(A,M, 0.00001)
			[test_val, test_vect] = cholesky_wilkinson(A,M)
			err_1 = average_error(A, M , eigenval, eigenvect)
			err_2 = average_error(A, M, test_val, test_vect)
			fp.write(",{},{}".format(err_1, err_2))
		fp.write("\n")
	fp.close()

