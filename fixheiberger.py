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
		#print "Case 1"
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
			
			#print "Case 2"
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

			#print "Case 3"

			# General case: Possibly singular A_13
			# TODO Assert n_1 < dim - n_1 - n_3
			A_13 = A[0:n_1, n_1 + n_3: dim]
			n_4 = dim - n_1 - n_3
			if n_1 < n_4: 
				print "WARNING: EXCESSIVE TOLERANCE RESTRICTIONS IN n_4"
			[Q_11,A_14] = linalg.qr(A_13, mode = "full")
			Q = np.matrix(linalg.block_diag(Q_11, np.identity(dim - n_1)))

			A = Q.getH() * A * Q
			# Solve the equivalent problem
			B_22 = A[n_4: n_1, n_4: n_1]
			B_23 = A[n_4: n_1, n_1: n_1 + n_3]
			Psi_inv = np.diag(1/z[0:n_3])
			[eigenval, x_2] = linalg.eigh(B_22 - B_23 * Psi_inv * B_23.getH())

			# Reconstruct Eigenvectors
			x_1 = np.zeros((n_4, len(eigenval)))
			x_3 = -Psi_inv* B_23.getH() * x_2

			B_12 = A[0:n_4, n_4: n_1]
			B_13 = A[0:n_4, n_1: n_1 + n_3]
			B_14 = A[0:n_4, n_1 + n_3:dim]
			x_4 = linalg.solve_triangular(B_14, -(B_12*x_2 + B_13*x_3))

			eigenvect = np.concatenate((x_1, x_2, x_3, x_4), axis = 0)
			return eigenval, G * D * U *Q *eigenvect


#  Unit Testing =============================
if __name__ == "__main__":

	def run_test(A,M, r):

		[fh_val, fh_vect] = fix_heiberger(A,M,r)
		[def_val, def_vect] = linalg.eigh(A,M)
		[cw_val, cw_vect] = cholesky_wilkinson(A,M)
		print "Fix-Heiberger with error: ", average_error(A,M, fh_val, fh_vect)
		print "Cholesky-Wilkinson with error: ", average_error(A,M, cw_val, cw_vect)
		print "Scipy with error: ", average_error(A,M, def_val, def_vect),  "\n"

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
		M = matgen.rand_by_eigenrange(100*i, 0,100000000)
		run_test(A,M, 0.1)
