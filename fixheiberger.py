import numpy as np
import numpy.matlib
from scipy import linalg
from choleskywilkinson import cholesky_wilkinson
import matrixgenerator as matgen

# Preliminary, unoptimized implementation of Fix-Heiberger Algorithm

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


# Helper Functions ==================================================
def find_contraction_index(eigenval, r):
	a, b, index = 0, sum(eigenval), 0
	for i in range(len(eigenval)):
		index = i
		if abs(eigenval[i]) < r * abs(eigenval[0]):
			return index
	index = index + 1
	return index

def sort_eigenval(eigenvect, eigenval, p = False): 

	def getkey(item):
		return abs(item[0])

	tuples = []
	for i in range(len(eigenval)):
		tuples.append([eigenval[i],i])
	tuples = sorted(tuples, key = getkey, reverse = True)

	G = np.zeros((len(eigenval), len(eigenval)))
	for i in range(len(eigenval)):
		eigenval[i] = tuples[i][0]
		G[:,i] = eigenvect[:, tuples[i][1]]
	return np.matrix(G)

	# Construct array of tupes
	# Comparator function
	# Construct new eigenvect G matrix


# Main Algorithm =====================================================

def fix_heiberger(A,M, r, cond = True):

	# Spectral Decompose M. Find G -> Use SVD or Schur?
	# [D,G] = linalg.schur(M) - Does not sort by eigenvalues.
	[d,G] = linalg.eigh(M)
	dim = len(d)
	G = sort_eigenval(G,d)

	#I = np.matlib.identity(dim)
	#print "Step 1 Error:", average_error(M, I, d, G)

	# Deflation Procedure M
	n_1 = find_contraction_index(d, r)
	d[n_1:] = [0 for e in d[n_1:]]
	A = G.getH()* A * G
	
	d_root_inv = 1/np.sqrt(d[0:n_1])
	D = np.matrix(np.diag(d_root_inv))
	D = linalg.block_diag(D, np.identity(dim - n_1))
	A = D*A*D 

	if (n_1> dim - 1 ):
		
		# If M is well-conditioned, simply solve the newly decomposed problem.
		print "Case 1"
		[eigenval, eigenvect] = linalg.eigh(A)

		# Reconstruct eigenvectors
		return eigenval, G * D * eigenvect

	else:
		
		# Spectral Decompose submatrix of A
		# Deflation Proedure on A_22
		A_22 = A[n_1:dim, n_1:dim]
		[z,F] = linalg.eigh(A_22)
		F = sort_eigenval(F,z, True)
		n_3 = find_contraction_index(z, r)

		z[n_3:] = [0 for e in z[n_3:]]

		U = np.matrix(linalg.block_diag(np.identity(n_1), F))
		A = U.getH()*A*U
		A[n_1: , n_1: ] = np.diag(z)

		if (n_3 > len(z)-1):
			
			print "Case 2"
			# When A_22 is well-conditioned
			A_12 = A[0:n_1, n_1:dim]
			A_11 = A[0:n_1, 0:n_1]
			Psi_inv = np.diag(1/z)
			[eigenval, x_1] = linalg.eigh(A_11 - A_12 * Psi_inv * A_12.getH())

			# Reconstruct eigenvectors
			x_2 = - Psi_inv * A_12.getH() * np.matrix(x_1)
			eigenvect = np.concatenate((x_1,x_2), axis=0)
			return eigenval, G * D * U * eigenvect

		else:

			print "Case 3"

			# General case: Possibly singular A_13
			# TODO Assert n_1 < dim - n_1 - n_3
			A_13 = A[0:n_1, n_1 + n_3: dim]
			n_4 = dim - n_1 - n_3
			if n_1 < n_4: 
				print "WARNING: EXCESSIVE TOLERANCE RESTRICTIONS IN n_4"
			[Q_11,A_14] = linalg.qr(A_13, mode = "economic")
			Q = np.matrix(linalg.block_diag(Q_11, np.identity(dim - n_1)))

			A = Q.getH() * A * Q

			

			'''[Q,s,Ph] = linalg.svd(A_13)
			index_b = find_contraction_index(s,r)
			s[index_b:] = [0 for e in s[index_b:]]
			I = np.matlib.identity(index_a)
			V = np.matrix(linalg.block_diag(Q, I, (np.matrix(Ph)).getH()))
			
			A = V * A * V.getH()
			A[0:index, index+index_a:dim] = np.zeros((index, dim - index - index_a))
			A[0:len(s), index+index_a: index+index_a+len(s)] = np.diag(s)
			A[index+index_a:dim, 0:index] = np.matrix(A[0:index, index+index_a:dim]).getH()

			# Solve the equivalent problem
			B_22 = A[index_b: index, index_b: index]
			B_23 = A[index_b: index, index: index+index_a]
			Psi_inv = np.diag(1/z[0:index_a])
			[eigenval, x_2] = linalg.eigh(B_22 - B_23 * Psi_inv * B_23.getH())

			# Reconstruct Eigenvectors
			x_1 = np.zeros((index_b, len(eigenval)))

			z_inv = 1/z[0:index_a]
			x_3 = -np.diag(z_inv)* B_23.getH() * x_2

			s_inv = 1/s[0:index_b]
			B_12 = A[0:index_b, index_b: index]
			B_13 = A[0:index_b, index: index + index_a]
			x_4 = -np.diag(s_inv) * (B_12*x_2 + B_13*x_3)

			x_5 = np.zeros((dim - index - index_a - index_b, len(eigenval)))

			eigenvect = np.concatenate((x_1, x_2, x_3, x_4, x_5), axis = 0)
			return eigenval, V * U * G * eigenvect'''


#  Unit Testing =============================
if __name__ == "__main__":

	def run_test(A,M, r):

		[fh_val, fh_vect] = fix_heiberger(A,M,r)
		[def_val, def_vect] = linalg.eigh(A,M)
		[cw_val, cw_vect] = cholesky_wilkinson(A,M)
		print "Fix-Heiberger with error: ", average_error(A,M, fh_val, fh_vect)
		print "Cholesky-Wilkinson with error: ", average_error(A,M, cw_val, cw_vect)
		print "Scipy with error: ", average_error(A,M, def_val, def_vect), "\n"

	# Testing for Case 1 Performance
	print "Testing Case 1 \n"
	for i in range(2,8):
		A = matgen.rand_symm(20*i)
		M = matgen.rand_by_eigenrange(20*i, 1,100000)
		run_test(A,M,0.0000011)

	# Testing for Case 2 Performance

	print "Testing Case 2\n"
	for i in range(2,8):
		A = matgen.rand_symm(100*i)
		M = matgen.rand_by_eigenrange(100*i, 0,10000000)
		run_test(A,M, 0.01)

	print "Testing Case 3\n"
	for i in range(2,8):
		A = matgen.rand_symm(100*i)
		M = matgen.rand_by_eigenrange(100*i, 0,10000000)
		run_test(A,M, 0.1)
