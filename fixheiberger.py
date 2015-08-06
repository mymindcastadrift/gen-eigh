import numpy as np
import numpy.matlib
from scipy import linalg
from choleskywilkinson import cholesky_wilkinson
import matrixgenerator as matgen

# Preliminary, unoptimized implementation of Fix-Heiberger Algorithm

# Helper Functions ==================================================
def find_contraction_index(eigenval, r):
	a, b, index = 0, sum(eigenval), 0
	for i in range(len(eigenval)):
		index = i
		if abs(eigenval[i]) < r * abs(eigenval[0]):
			return index
	index = index + 1
	return index

def sort_eigenval(eigenvect, eigenval): 

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

	# Deflation Procedure M
	index = find_contraction_index(d, r)
	d[index:] = [0 for e in d[index:]]
	A = G.getH()* A * G
	d_root_inv = 1/np.sqrt(d[0:index])

	if (index > dim - 1 ):
		
		# If M is well-conditioned, simply solve the newly decomposed problem.
		print "Case 1"
		F = np.matlib.identity(dim-index)
		U = np.matrix(linalg.block_diag(np.diag(d_root_inv), F))
		Z = U.getH()*A*U
		[eigenval, eigenvect] = linalg.eigh(Z)

		# Reconstruct eigenvectors
		return eigenval, G * eigenvect

	else:
		
		# Spectral Decompose submatrix of A
		# Deflation Proedure on A_22
		A_22 = A[index:dim, index:dim]
		[z,F] = linalg.eigh(A_22)
		F = sort_eigenval(F,z)
		index_a = find_contraction_index(z, r)

		z[index_a:] = [0 for e in z[index_a:]]

		U = np.matrix(linalg.block_diag(np.diag(d_root_inv), F))
		Z = U.getH()*A*U
		Z[index: , index: ] = np.diag(z)

		if (index_a > A_22.shape[0]-1):
			
			print "Case 2"
			# When A_22 is well-conditioned
			A_12 = Z[0:index, index:dim]
			A_11 = Z[0:index, 0:index]
			Psi_inv = np.diag(1/z)
			[eigenval, x_1] = linalg.eigh(A_11 - A_12 * Psi_inv * A_12.getH())

			# Reconstruct eigenvectors
			z_inv = 1/z[0:index_a]
			x_2 = -np.diag(z_inv)* A_12.getH() * x_1
			eigenvect = np.concatenate((x_1,x_2), axis=0)
			return eigenval, U * G * eigenvect

		else:

			print "Case 3"

			# General case: Possibly singular A_13
			A_13 = Z[0:index, index + index_a: dim]
			#print "A_13:", A_13

			[Q,s,Ph] = linalg.svd(A_13)
			index_b = find_contraction_index(s,r)
			s[index_b:] = [0 for e in s[index_b:]]
			I = np.matlib.identity(index_a)
			V = np.matrix(linalg.block_diag(Q, I, (np.matrix(Ph)).getH()))
			
			B = V * Z * V.getH()
			B[0:index, index+index_a:dim] = np.zeros((index, dim - index - index_a))
			B[0:len(s), index+index_a: index+index_a+len(s)] = np.diag(s)
			B[index+index_a:dim, 0:index] = np.matrix(B[0:index, index+index_a:dim]).getH()

			# Solve the equivalent problem
			B_22 = B[index_b: index, index_b: index]
			B_23 = B[index_b: index, index: index+index_a]
			Psi_inv = np.diag(1/z[0:index_a])
			[eigenval, x_2] = linalg.eigh(B_22 - B_23 * Psi_inv * B_23.getH())

			# Reconstruct Eigenvectors
			x_1 = np.zeros((index_b, len(eigenval)))

			z_inv = 1/z[0:index_a]
			x_3 = -np.diag(z_inv)* B_23.getH() * x_2

			s_inv = 1/s[0:index_b]
			B_12 = B[0:index_b, index_b: index]
			B_13 = B[0:index_b, index: index + index_a]
			x_4 = -np.diag(s_inv) * (B_12*x_2 + B_13*x_3)

			x_5 = np.zeros((dim - index - index_a - index_b, len(eigenval)))

			eigenvect = np.concatenate((x_1, x_2, x_3, x_4, x_5), axis = 0)
			return eigenval, V * U * G * eigenvect


#  Unit Testing =============================
if __name__ == "__main__":

	def run_test(A,M, r):

		[fh_val, fh_vect] = fix_heiberger(A,M,r)
		[cw_val, cw_vect] = cholesky_wilkinson(A,M)
		print "Fix-Heiberger: ", fh_val, " with error: ", average_error(A,M, fh_val, fh_vect)
		print "Cholesky-Wilkinson: ", cw_val, " with error: ", average_error(A,M, cw_val, cw_vect)

	def test_correct(n):

		print "Testing with dimensions:", n

		z = range(1,n+1)
		z = [i+0.01 for i in z]
		A = matgen.rand_by_eigenval(n, z[::-1])
		M = matgen.diag([1]*n)
		
		run_test(A,M,0.01)

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

	test_correct(100)