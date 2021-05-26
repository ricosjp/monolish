#pragma once
#include "../../../include/common/monolish_common.hpp"
#include <stdio.h>
#include <vector>

#if defined USE_MPI
#include <mpi.h>
#endif

/**
 * @brief
 * Linear Algebra Package for Dense Matrix
 */
namespace monolish::internal::lapack {

//////////////////////////////////////////////////////
//  Eigenvalue calculation
//////////////////////////////////////////////////////

/**
 * @brief Finds all the eigenvalues and eigenvectors of a single precision dense
 * symmetric matrix A using Divide and Conquer method
 * @param A single precision symmetric Dense matrix (size M x M)
 * when jobz is 'V' and returning 0, A becomes the orthonormal eigenvectors
 * when jobz is 'N' and uplo is 'L', the upper triangle including the diagonal
 * of the matrix A is overwritten
 * when jobz is 'N' and uplo is 'U', the upper triangle including the diagonal
 * of the matrix A is overwritten
 * @param W single precision vector (size M)
 * when returning 0, W contains the eigenvalues of the matrix A
 * in ascending order
 * @param jobz when 'N' then calculate eigenvalues only, and when 'V' calculate
 * eigenvalues and eigenvectors
 * @param uplo when 'U' then upper triangle of A is used, and when 'L' then
 * lower triangle of A is used
 * @return 0 if successfully computed
 * -i then the i-th parameter had an illegal value
 * i then the algorithm failed to converge and i elements did not converge to 0
 * @note
 * - # of computation: M^3
 * - A is destroyed after called
 * - Multi-threading: true
 * - Temporary array is created inside the function
 * - GPU acceleration: false
 *    - # of data transfer: 0
 */
int syevd(matrix::Dense<float> &A, vector<float> &W, const char *jobz,
          const char *uplo);

/**
 * @brief Finds the eigenvalues and eigenvectors of a single precision dense
 * generalized symmetric definite eigenproblem of matrix A, B using Divide and
 * Conquer method
 * @param A single precision symmetric Dense matrix (size M x M)
 * when jobz is 'V' and returning 0, A becomes the orthonormal eigenvectors
 * when jobz is 'N' and uplo is 'L', the upper triangle including the diagonal
 * of the matrix A is overwritten
 * @param B single precision symmetric positive definite Dense matrix
 * (size M x M)
 * @param W single precision vector (size M)
 * when returning 0, W contains the eigenvalues of the eigenproblem
 * in ascending order
 * @param itype must be 1,2,3
 * if 1, A*x = lambda*B*x
 * if 2, A*B*x = lambda*x
 * if 3, B*A*x = lambda*x
 * @param jobz when 'N' then calculate eigenvalues only, and when 'V' calculate
 * eigenvalues and eigenvectors
 * @param uplo when 'U' then upper triangle of A, B is used, and when 'L' then
 * lower triangle of A, B is used
 * @return 0 if successfully computed
 * -i then the i-th parameter had an illegal value
 * i then the algorithm failed to converge and i elements did not converge to 0
 * @note
 * - # of computation: M^3
 * - A is destroyed after called
 * - Multi-threading: true
 * - Temporary array is created inside the function
 * - GPU acceleration: false
 *    - # of data transfer: 0
 */
int sygvd(matrix::Dense<float> &A, matrix::Dense<float> &B, vector<float> &W,
          const int itype, const char *jobz, const char *uplo);

//////////////////////////////////////////////////////
//  Linear equation
//////////////////////////////////////////////////////

/**
 * @brief LU bunkai...atode kaku
 * @param A single precision Dense matrix (size M x N)
 * @param ipiv integer array (size min(M,N))
 * @return 0 if successfully computed
 * -i then the i-th parameter had an illegal value
 * i then the algorithm failed to converge and i elements did not converge to 0
 * @note
 * - # of computation: XXXXX
 * - A is destroyed after called
 * - Multi-threading: true
 * - Temporary array is created inside the function
 * - GPU acceleration: ture
 *    - # of data transfer: min(M,N) integer array
 */
int getrf(matrix::Dense<float> &A, std::vector<int> &ipiv);

/**
 * @brief slove Lu...atode kaku
 * @param A single precision Dense matrix (size M x N)
 * @param B single precision vector (size M)
                     On entry, the right hand side matrix B. On exit, the
 solution matrix X.
 * @param ipiv integer array (size min(M,N))
 * @return 0 if successfully computed
 * -i then the i-th parameter had an illegal value
 * i then the algorithm failed to converge and i elements did not converge to 0
 * @note
 * - # of computation: XXXXX
 * - A is destroyed after called
 * - Multi-threading: true
 * - Temporary array is created inside the function
 * - GPU acceleration: ture
 *    - # of data transfer: min(M,N) integer array
 */
int getrs(const matrix::Dense<float> &A, vector<float> &B,
          const std::vector<int> &ipiv);

/**
 * @brief LU bunkai...atode kaku
 * @param A single precision symmetric Dense matrix (size M x M)
 * @param ipiv integer array (size min(M,N))
 * @return 0 if successfully computed
 * -i then the i-th parameter had an illegal value
 * i then the algorithm failed to converge and i elements did not converge to 0
 * @note
 * - # of computation: XXXXX
 * - A is destroyed after called
 * - Multi-threading: true
 * - Temporary array is created inside the function
 * - GPU acceleration: ture
 *    - # of data transfer: min(M,N) integer array
 */
int sytrf(matrix::Dense<float> &A, std::vector<int> &ipiv);

/**
 * @brief slove Lu...atode kaku
 * @param A single precision symmetric Dense matrix (size M x M)
 * @param B single precision vector (size M)
                     On entry, the right hand side matrix B. On exit, the
 solution matrix X.
 * @param ipiv integer array (size min(M,N))
 * @return 0 if successfully computed
 * -i then the i-th parameter had an illegal value
 * i then the algorithm failed to converge and i elements did not converge to 0
 * @note
 * - # of computation: XXXXX
 * - A is destroyed after called
 * - Multi-threading: true
 * - Temporary array is created inside the function
 * - GPU acceleration: ture
 *    - # of data transfer: min(M,N) integer array
 */
int sytrs(const matrix::Dense<float> &A, vector<float> &B,
          const std::vector<int> &ipiv);

} // namespace monolish::internal::lapack
