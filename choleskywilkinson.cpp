/* Choleksy-Wilkinson Implementation with C++ and Accelerate Library (Fortran)
** Compile with g++ --std=c++11 choleskywilkinson.cpp -o choleskywilkinson -framework Accelerate
*/

#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <ctime>

#define FUNDERSCORE
#ifdef FUNDERSCORE
# define BLAS(name) name ## _
#else
# define BLAS(name) name
#endif


// BLAS function symbols ===============================

extern "C" 
{
	void BLAS(dgemm)
	( const char* transa, const char* transb,
	  const int* m, const int* n, const int* k,
	  const double* alpha, const double* A, const int* ldA,
	                       const double* B, const int* ldB,
	  const double* beta,        double* C, const int* ldC 
	 );

	void BLAS(dtrsm)
	(
		const char* side, const char* uplo, const char* transa, const char* diag,
		const int* M, const int* N, const double* alpha, const double* A, 
		const int* lda, const double* B, const int* ldb
	);

	void BLAS(dpotrf)
	(
		const char* uplo, const int* N, const double* A, const int* lda, const int* output
	);

	void BLAS(dsyev)
	(
		const char* jobz, const char* uplo, const int* N, const double* A, const int* lda,
		const double* W, const double* work, const int* lwork, const int* info
	);
}


// Helper functions ======================

template<typename T>
void Print
( const std::vector<T>& A, int m, int n, const std::string msg="matrix" )
{
    std::cout << msg << std::endl;
    for( int i=0; i<m; ++i )
    {
        for( int j=0; j<n; ++j )
            std::cout << A[i+j*m] << " ";
        std::cout << "\n";
    }
    std::cout << std::endl;
}

std::vector<double> generate_random_matrix(const int m, const int n, const double lower, const double upper)
{
	std::default_random_engine generator(time(0));
	std::uniform_real_distribution<double> distribution(lower, upper);

	std::vector<double> A(m*n);
	for(int i=0; i<m*n; i++){
		A[i] = distribution(generator);
	}

	return A;
}

/*double error(std::vector<double> *A, std::vector<double> *M, int N, std::vector<double> *eigenval, std::vector<double> *eigenvect){
	double err;
	vector<double> err_array(N);

	return err;
}*/

// Cholesky-Wilkinson Function =====================
int choleskywilkinson(std::vector<double> A, std::vector<double> M, int N, std::vector<double> *eigenval, std::vector<double> *eigenvect)
{
	// Cholesky Factorization of M = LL*. L is stored in M;
	char uplo = 'L';
	int info;
	BLAS(dpotrf)( &uplo, &N, M.data(), &N, &info );

	Print (M, N, N, "L is");

	// Triangular solve to obtain G = L-1 A L-*
	// (1) Solve S s.t. LX = A
	// (2) Solve Y s.t. YL* = S
	char side = 'L', diag = 'N', transA = 'N';
	uplo = 'L';
	double alpha = 1; 
	BLAS(dtrsm)( &side, &uplo, &transA, &diag, &N, &N, &alpha, M.data(), &N, A.data(), &N );

	side = 'R'; transA = 'T';
	BLAS(dtrsm)( &side, &uplo, &transA, &diag, &N, &N, &alpha, M.data(), &N, A.data(), &N );

	Print(A, N, N, "G is");

	// Eigenvalue problem GZ = lambda Z
	char jobz = 'V';
	int lwork = -1;
	double lwork_dbl;
	BLAS(dsyev)( &jobz, &uplo, &N, A.data(), &N, eigenval->data(), &lwork_dbl, &lwork, &info );
	lwork = (int)lwork_dbl;
	std::vector<double> work(lwork);
	BLAS(dsyev)( &jobz, &uplo, &N, A.data(), &N, eigenval->data(), work.data(), &lwork, &info );

	Print(A, N, N, "modified eigenvect is");

	// Triangular solve to obtain original eigenvalues: L*X = Z
	side = 'L'; uplo = 'L'; transA = 'N'; diag = 'N'; alpha = 1; 
	BLAS(dtrsm)( &side, &uplo, &transA, &diag, &N, &N, &alpha, M.data(), &N, A.data(), &N );

	Print (*eigenval, 1, N, "eigenval is");
	Print(A, N, N, "eigenvect is");

	*eigenvect = A;
	return 0;
}


// Unit Testing ==================================================

int main(int argc, char** argv){

	// Random matrices
	const int N = 3;
	std::vector<double> temp(N*N), A(N*N), M(N*N);
	
	// Generate Symmetric A
	temp = generate_random_matrix(N, N, -100.0, 100.0);
	for (int i = 0; i< N; i++)
		for (int j=0; j<N; j++)
			A[i*N + j] = temp[i*N + j] + temp[j*N + i];

	// Generate Symmetric Positive Definite M
	temp = generate_random_matrix(N, N, -100.0, 100.0);
	char transA = 'T', transB = 'N';
	double alpha = 1, beta = 1;
	BLAS(dgemm)( &transA, &transB, &N, &N, &N, &alpha, temp.data(), &N, temp.data(), &N, &beta, M.data(), &N );

	Print(A, N, N, "A is");
	Print(M, N, N, "M is");

	// Calling Cholesky-Wilkinson
	std::vector<double> eigenval(N);
	std::vector<double> eigenvect(N*N);
	choleskywilkinson(A, M, N, &eigenval, &eigenvect);



	return 0;

}