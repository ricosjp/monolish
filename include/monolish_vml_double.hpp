#pragma once
#include "common/monolish_common.hpp"
#include <stdio.h>
#include <vector>

#if defined USE_MPI
#include <mpi.h>
#endif

namespace monolish {
/**
 * @brief
 * Vector and Matrix element-wise math library
 */
namespace vml {
//////////////////////////////////////////////////////
// Dense
//////////////////////////////////////////////////////
/**
 * @brief double precision element by element addition of Dense matrix A and
 * Dense matrix B.
 * @param A double precision monolish Dense Matrix (size M x N)
 * @param B double precision monolish Dense Matrix (size M x N)
 * @param C double precision monolish Dense Matrix (size M x N)
 * @note
 * - # of computation: MN
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
void add(const matrix::Dense<double> &A, const matrix::Dense<double> &B,
         matrix::Dense<double> &C);

/**
 * @brief double precision element by element subtraction of Dense matrix A and
 * Dense matrix B.
 * @param A double precision monolish Dense Matrix (size M x N)
 * @param B double precision monolish Dense Matrix (size M x N)
 * @param C double precision monolish Dense Matrix (size M x N)
 * @note
 * - # of computation: MN
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
void sub(const matrix::Dense<double> &A, const matrix::Dense<double> &B,
         matrix::Dense<double> &C);

/**
 * @brief double precision element by element multiplication of Dense matrix A
 * and Dense matrix B.
 * @param A double precision monolish Dense Matrix (size M x N)
 * @param B double precision monolish Dense Matrix (size M x N)
 * @param C double precision monolish Dense Matrix (size M x N)
 * @note
 * - # of computation: MN
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
void mul(const matrix::Dense<double> &A, const matrix::Dense<double> &B,
         matrix::Dense<double> &C);

/**
 * @brief double precision element by element division of Dense matrix A and
 * Dense matrix B.
 * @param A double precision monolish Dense Matrix (size M x N)
 * @param B double precision monolish Dense Matrix (size M x N)
 * @param C double precision monolish Dense Matrix (size M x N)
 * @note
 * - # of computation: MN
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
void div(const matrix::Dense<double> &A, const matrix::Dense<double> &B,
         matrix::Dense<double> &C);

/**
 * @brief double precision scalar and Dence matrix addition (C[i][j] = A[i][j] +
 * alpha)
 * @param A double precision monolish Dense Matrix (size M x N)
 * @param alpha double precision scalar value
 * @param C double precision monolish Dense Matrix (size M x N)
 * @note
 * - # of computation: MN
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
void add(const matrix::Dense<double> &A, const double alpha,
         matrix::Dense<double> &C);

/**
 * @brief double precision scalar and Dence matrix subtraction (C[i][j] =
 * A[i][j] + alpha)
 * @param A double precision monolish Dense Matrix (size M x N)
 * @param alpha double precision scalar value
 * @param C double precision monolish Dense Matrix (size M x N)
 * @note
 * - # of computation: MN
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
void sub(const matrix::Dense<double> &A, const double alpha,
         matrix::Dense<double> &C);

/**
 * @brief double precision scalar and Dence matrix multiplication (C[i][j] =
 * A[i][j] + alpha)
 * @param A double precision monolish Dense Matrix (size M x N)
 * @param alpha double precision scalar value
 * @param C double precision monolish Dense Matrix (size M x N)
 * @note
 * - # of computation: MN
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
void mul(const matrix::Dense<double> &A, const double alpha,
         matrix::Dense<double> &C);

/**
 * @brief double precision scalar and Dence matrix division (C[i][j] = A[i][j] +
 * alpha)
 * @param A double precision monolish Dense Matrix (size M x N)
 * @param alpha double precision scalar value
 * @param C double precision monolish Dense Matrix (size M x N)
 * @note
 * - # of computation: MN
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
void div(const matrix::Dense<double> &A, const double alpha,
         matrix::Dense<double> &C);

/**
 * @brief power to double precision dense matrix elements by double precision
 *dense matrix (C[0:N] = pow(A[0:N], B[0:N]))
 * @param A double precision monolish Dense Matrix (size M x N)
 * @param B double precision monolish Dense Matrix (size M x N)
 * @param C double precision monolish Dense Matrix (size M x N)
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 **/
void pow(const matrix::Dense<double> &A, const matrix::Dense<double> &B,
         matrix::Dense<double> &C);

/**
 * @brief power to double precision dense matrix elements by double precision
 *scalar value (C[0:N] = pow(A[0:N], alpha))
 * @param A double precision monolish Dense Matrix (size M x N)
 * @param alpha double precision scalar value
 * @param C double precision monolish Dense Matrix (size M x N)
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 **/
void pow(const matrix::Dense<double> &A, const double alpha,
         matrix::Dense<double> &C);

/**
 * @brief sqrt to double precision dense matrix elements (C[0:N] = sqrt(A[0:N]))
 * @param A double precision monolish Dense Matrix (size M x N)
 * @param C double precision monolish Dense Matrix (size M x N)
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 **/
void sqrt(const matrix::Dense<double> &A, matrix::Dense<double> &C);

/**
 * @brief sin to double precision dense matrix elements (C[0:N] = sin(A[0:N]))
 * @param A double precision monolish Dense Matrix (size M x N)
 * @param C double precision monolish Dense Matrix (size M x N)
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 **/
void sin(const matrix::Dense<double> &A, matrix::Dense<double> &C);

/**
 * @brief sinh to double precision dense matrix elements (C[0:N] = sinh(A[0:N]))
 * @param A double precision monolish Dense Matrix (size M x N)
 * @param C double precision monolish Dense Matrix (size M x N)
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 **/
void sinh(const matrix::Dense<double> &A, matrix::Dense<double> &C);

/**
 * @brief asin to double precision dense matrix elements (C[0:N] = asin(A[0:N]))
 * @param A double precision monolish Dense Matrix (size M x N)
 * @param C double precision monolish Dense Matrix (size M x N)
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 **/
void asin(const matrix::Dense<double> &A, matrix::Dense<double> &C);

/**
 * @brief asinh to double precision dense matrix elements (C[0:N] =
 *asinh(A[0:N]))
 * @param A double precision monolish Dense Matrix (size M x N)
 * @param C double precision monolish Dense Matrix (size M x N)
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 **/
void asinh(const matrix::Dense<double> &A, matrix::Dense<double> &C);

/**
 * @brief tan to double precision dense matrix elements (C[0:N] = tan(A[0:N]))
 * @param A double precision monolish Dense Matrix (size M x N)
 * @param C double precision monolish Dense Matrix (size M x N)
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 **/
void tan(const matrix::Dense<double> &A, matrix::Dense<double> &C);

/**
 * @brief tanh to double precision dense matrix elements (C[0:N] = tanh(A[0:N]))
 * @param A double precision monolish Dense Matrix (size M x N)
 * @param C double precision monolish Dense Matrix (size M x N)
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 **/
void tanh(const matrix::Dense<double> &A, matrix::Dense<double> &C);

/**
 * @brief atan to double precision dense matrix elements (C[0:N] = atan(A[0:N]))
 * @param A double precision monolish Dense Matrix (size M x N)
 * @param C double precision monolish Dense Matrix (size M x N)
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 **/
void atan(const matrix::Dense<double> &A, matrix::Dense<double> &C);

/**
 * @brief atanh to double precision dense matrix elements (C[0:N] =
 *atanh(A[0:N]))
 * @param A double precision monolish Dense Matrix (size M x N)
 * @param C double precision monolish Dense Matrix (size M x N)
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 **/
void atanh(const matrix::Dense<double> &A, matrix::Dense<double> &C);

/**
 * @brief ceil to double precision dense matrix elements (C[0:N] = ceil(A[0:N]))
 * @param A double precision monolish Dense Matrix (size M x N)
 * @param C double precision monolish Dense Matrix (size M x N)
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 **/
void ceil(const matrix::Dense<double> &A, matrix::Dense<double> &C);

/**
 * @brief floor to double precision dense matrix elements (C[0:N] =
 *floor(A[0:N]))
 * @param A double precision monolish Dense Matrix (size M x N)
 * @param C double precision monolish Dense Matrix (size M x N)
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 **/
void floor(const matrix::Dense<double> &A, matrix::Dense<double> &C);

/**
 * @brief sign inversion to double precision dense matrix elements (C[0:N] =
 *sign(A[0:N]))
 * @param A double precision monolish Dense Matrix (size M x N)
 * @param C double precision monolish Dense Matrix (size M x N)
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 **/
void sign(const matrix::Dense<double> &A, matrix::Dense<double> &C);

/**
 * @brief reciprocal to double precision dense matrix elements (C[0:N] =
 * 1 / A[0:N])
 * @param A double precision monolish Dense Matrix (size M x N)
 * @param C double precision monolish Dense Matrix (size M x N)
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 **/
void reciprocal(const matrix::Dense<double> &A, matrix::Dense<double> &C);

/**
 * @brief Create a new matrix with greatest elements of two matrices (C[0:nnz] =
 *A(a[0:nnz], B[0:nnz]))
 * @param A double precision monolish Dense matrix (size M x N)
 * @param B double precision monolish Dense matrix (size M x N)
 * @param C double precision monolish Dense matrix (size M x N)
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 **/
void max(const matrix::Dense<double> &A, const matrix::Dense<double> &B,
         matrix::Dense<double> &C);

/**
 * @brief Create a new matrix with smallest elements of two matrices (C[0:nnz] =
 *A(a[0:nnz], B[0:nnz]))
 * @param A double precision monolish Dense matrix (size M x N)
 * @param B double precision monolish Dense matrix (size M x N)
 * @param C double precision monolish Dense matrix (size M x N)
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 **/
void min(const matrix::Dense<double> &A, const matrix::Dense<double> &B,
         matrix::Dense<double> &C);

/**
 * @brief Finds the greatest element in double precision Dense matrices
 *(max_element(C[0:nnz]))
 * @param C double precision monolish Dense matrix (size M x N)
 * @return greatest value
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 **/
double max(const matrix::Dense<double> &C);
/**
 * @brief Finds the greatest element in double precision Dense matrices
 *(max_element(C[0:nnz]))
 * @param C double precision monolish Dense matrix (size M x N)
 * @return greatest value
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 **/
double min(const matrix::Dense<double> &C);
//////////////////////////////////////////////////////
// CRS
//////////////////////////////////////////////////////

/**
 * @brief double precision element by element addition of CRS matrix A and CRS
 * matrix B.
 * @param A double precision monolish CRS Matrix (size M x N)
 * @param B double precision monolish CRS Matrix (size M x N)
 * @param C double precision monolish CRS Matrix (size M x N)
 * @note
 * - # of computation: MN
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 * @warning
 * A, B, and C must be same non-zero structure
 */
void add(const matrix::CRS<double> &A, const matrix::CRS<double> &B,
         matrix::CRS<double> &C);

/**
 * @brief double precision element by element subtraction of CRS matrix A and
 * CRS matrix B.
 * @param A double precision monolish CRS Matrix (size M x N)
 * @param B double precision monolish CRS Matrix (size M x N)
 * @param C double precision monolish CRS Matrix (size M x N)
 * @note
 * - # of computation: MN
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 * @warning
 * A, B, and C must be same non-zero structure
 */
void sub(const matrix::CRS<double> &A, const matrix::CRS<double> &B,
         matrix::CRS<double> &C);

/**
 * @brief double precision element by element multiplication of CRS matrix A and
 * CRS matrix B.
 * @param A double precision monolish CRS Matrix (size M x N)
 * @param B double precision monolish CRS Matrix (size M x N)
 * @param C double precision monolish CRS Matrix (size M x N)
 * @note
 * - # of computation: MN
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 * @warning
 * A, B, and C must be same non-zero structure
 */
void mul(const matrix::CRS<double> &A, const matrix::CRS<double> &B,
         matrix::CRS<double> &C);

/**
 * @brief double precision element by element division of CRS matrix A and CRS
 * matrix B.
 * @param A double precision monolish CRS Matrix (size M x N)
 * @param B double precision monolish CRS Matrix (size M x N)
 * @param C double precision monolish CRS Matrix (size M x N)
 * @note
 * - # of computation: MN
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 * @warning
 * A, B, and C must be same non-zero structure
 */
void div(const matrix::CRS<double> &A, const matrix::CRS<double> &B,
         matrix::CRS<double> &C);

/**
 * @brief double precision scalar and Dence matrix addition (C[i][j] = A[i][j] +
 * alpha)
 * @param A double precision monolish CRS Matrix (size M x N)
 * @param alpha double precision scalar value
 * @param C double precision monolish CRS Matrix (size M x N)
 * @note
 * - # of computation: MN
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 * @warning
 * A and C must be same non-zero structure
 */
void add(const matrix::CRS<double> &A, const double alpha,
         matrix::CRS<double> &C);

/**
 * @brief double precision scalar and Dence matrix subtraction (C[i][j] =
 * A[i][j] + alpha)
 * @param A double precision monolish CRS Matrix (size M x N)
 * @param alpha double precision scalar value
 * @param C double precision monolish CRS Matrix (size M x N)
 * @note
 * - # of computation: MN
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 * @warning
 * A and C must be same non-zero structure
 */
void sub(const matrix::CRS<double> &A, const double alpha,
         matrix::CRS<double> &C);

/**
 * @brief double precision scalar and Dence matrix multiplication (C[i][j] =
 * A[i][j] + alpha)
 * @param A double precision monolish CRS Matrix (size M x N)
 * @param alpha double precision scalar value
 * @param C double precision monolish CRS Matrix (size M x N)
 * @note
 * - # of computation: MN
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 * @warning
 * A and C must be same non-zero structure
 */
void mul(const matrix::CRS<double> &A, const double alpha,
         matrix::CRS<double> &C);

/**
 * @brief double precision scalar and Dence matrix division (C[i][j] = A[i][j] +
 * alpha)
 * @param A double precision monolish CRS Matrix (size M x N)
 * @param alpha double precision scalar value
 * @param C double precision monolish CRS Matrix (size M x N)
 * @note
 * - # of computation: MN
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 * @warning
 * A and C must be same non-zero structure
 */
void div(const matrix::CRS<double> &A, const double alpha,
         matrix::CRS<double> &C);

/**
 * @brief power to double precision CRS matrix elements by double precision CRS
 *matrix (C[0:N] = pow(A[0:N], b[0:N]))
 * @param A double precision monolish CRS Matrix (size M x N)
 * @param B double precision monolish CRS Matrix (size M x N)
 * @param C double precision monolish CRS Matrix (size M x N)
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 * @warning
 * A, B and C must be same non-zero structure
 **/
void pow(const matrix::CRS<double> &A, const matrix::CRS<double> &B,
         matrix::CRS<double> &C);

/**
 * @brief power to double precision CRS matrix elements by double precision
 *scalar value (C[0:N] = pow(A[0:N], alpha))
 * @param A double precision monolish CRS Matrix (size M x N)
 * @param alpha double precision scalar value
 * @param C double precision monolish CRS Matrix (size M x N)
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 * @warning
 * A and C must be same non-zero structure
 **/
void pow(const matrix::CRS<double> &A, const double alpha,
         matrix::CRS<double> &C);

/**
 * @brief sqrt to double precision CRS matrix elements (C[0:N] = sqrt(A[0:N]))
 * @param A double precision monolish CRS Matrix (size M x N)
 * @param B double precision monolish CRS Matrix (size M x N)
 * @param C double precision monolish CRS Matrix (size M x N)
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 * @warning
 * A, B and C must be same non-zero structure
 **/
void sqrt(const matrix::CRS<double> &A, matrix::CRS<double> &C);

/**
 * @brief sin to double precision CRS matrix elements (C[0:N] = sin(A[0:N]))
 * @param A double precision monolish CRS Matrix (size M x N)
 * @param B double precision monolish CRS Matrix (size M x N)
 * @param C double precision monolish CRS Matrix (size M x N)
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 * @warning
 * A, B and C must be same non-zero structure
 **/
void sin(const matrix::CRS<double> &A, matrix::CRS<double> &C);

/**
 * @brief sinh to double precision CRS matrix elements (C[0:N] = sinh(A[0:N]))
 * @param A double precision monolish CRS Matrix (size M x N)
 * @param B double precision monolish CRS Matrix (size M x N)
 * @param C double precision monolish CRS Matrix (size M x N)
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 * @warning
 * A, B and C must be same non-zero structure
 **/
void sinh(const matrix::CRS<double> &A, matrix::CRS<double> &C);

/**
 * @brief asin to double precision CRS matrix elements (C[0:N] = asin(A[0:N]))
 * @param A double precision monolish CRS Matrix (size M x N)
 * @param B double precision monolish CRS Matrix (size M x N)
 * @param C double precision monolish CRS Matrix (size M x N)
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 * @warning
 * A, B and C must be same non-zero structure
 **/
void asin(const matrix::CRS<double> &A, matrix::CRS<double> &C);

/**
 * @brief asinh to double precision CRS matrix elements (C[0:N] = asinh(A[0:N]))
 * @param A double precision monolish CRS Matrix (size M x N)
 * @param B double precision monolish CRS Matrix (size M x N)
 * @param C double precision monolish CRS Matrix (size M x N)
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 * @warning
 * A, B and C must be same non-zero structure
 **/
void asinh(const matrix::CRS<double> &A, matrix::CRS<double> &C);

/**
 * @brief tan to double precision CRS matrix elements (C[0:N] = tan(A[0:N]))
 * @param A double precision monolish CRS Matrix (size M x N)
 * @param B double precision monolish CRS Matrix (size M x N)
 * @param C double precision monolish CRS Matrix (size M x N)
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 * @warning
 * A, B and C must be same non-zero structure
 **/
void tan(const matrix::CRS<double> &A, matrix::CRS<double> &C);

/**
 * @brief tanh to double precision CRS matrix elements (C[0:N] = tanh(A[0:N]))
 * @param A double precision monolish CRS Matrix (size M x N)
 * @param B double precision monolish CRS Matrix (size M x N)
 * @param C double precision monolish CRS Matrix (size M x N)
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 * @warning
 * A, B and C must be same non-zero structure
 **/
void tanh(const matrix::CRS<double> &A, matrix::CRS<double> &C);

/**
 * @brief atan to double precision CRS matrix elements (C[0:N] = atan(A[0:N]))
 * @param A double precision monolish CRS Matrix (size M x N)
 * @param B double precision monolish CRS Matrix (size M x N)
 * @param C double precision monolish CRS Matrix (size M x N)
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 * @warning
 * A, B and C must be same non-zero structure
 **/
void atan(const matrix::CRS<double> &A, matrix::CRS<double> &C);

/**
 * @brief atanh to double precision CRS matrix elements (C[0:N] = atanh(A[0:N]))
 * @param A double precision monolish CRS Matrix (size M x N)
 * @param B double precision monolish CRS Matrix (size M x N)
 * @param C double precision monolish CRS Matrix (size M x N)
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 * @warning
 * A and C must be same non-zero structure
 **/
void atanh(const matrix::CRS<double> &A, matrix::CRS<double> &C);

/**
 * @brief ceil to double precision CRS matrix elements (C[0:N] = ceil(A[0:N]))
 * @param A double precision monolish CRS Matrix (size M x N)
 * @param B double precision monolish CRS Matrix (size M x N)
 * @param C double precision monolish CRS Matrix (size M x N)
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 * @warning
 * A and C must be same non-zero structure
 **/
void ceil(const matrix::CRS<double> &A, matrix::CRS<double> &C);

/**
 * @brief floor to double precision CRS matrix elements (C[0:N] = floor(A[0:N]))
 * @param A double precision monolish CRS Matrix (size M x N)
 * @param B double precision monolish CRS Matrix (size M x N)
 * @param C double precision monolish CRS Matrix (size M x N)
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 * @warning
 * A and C must be same non-zero structure
 **/
void floor(const matrix::CRS<double> &A, matrix::CRS<double> &C);

/**
 * @brief sign inversion to double precision CRS matrix elements (C[0:N] =
 *sign(A[0:N]))
 * @param A double precision monolish CRS Matrix (size M x N)
 * @param B double precision monolish CRS Matrix (size M x N)
 * @param C double precision monolish CRS Matrix (size M x N)
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 * @warning
 * A and C must be same non-zero structure
 **/
void sign(const matrix::CRS<double> &A, matrix::CRS<double> &C);

/**
 * @brief reciprocal to double precision dense matrix elements (C[0:N] =
 * 1 / A[0:N])
 * @param A double precision monolish CRS Matrix (size M x N)
 * @param C double precision monolish CRS Matrix (size M x N)
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 * @warning
 * A and C must be same non-zero structure
 **/
void reciprocal(const matrix::CRS<double> &A, matrix::CRS<double> &C);

/**
 * @brief Create a new matrix with greatest elements of two matrices (C[0:nnz] =
 *A(a[0:nnz], B[0:nnz]))
 * @param A double precision monolish CRS matrix (size M x N)
 * @param B double precision monolish CRS matrix (size M x N)
 * @param C double precision monolish CRS matrix (size M x N)
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 * @warning
 * A, B and C must be same non-zero structure
 **/
void max(const matrix::CRS<double> &A, const matrix::CRS<double> &B,
         matrix::CRS<double> &C);

/**
 * @brief Create a new matrix with smallest elements of two matrices (C[0:nnz] =
 *A(a[0:nnz], B[0:nnz]))
 * @param A double precision monolish CRS matrix (size M x N)
 * @param B double precision monolish CRS matrix (size M x N)
 * @param C double precision monolish CRS matrix (size M x N)
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 * @warning
 * A, B and C must be same non-zero structure
 **/
void min(const matrix::CRS<double> &A, const matrix::CRS<double> &B,
         matrix::CRS<double> &C);

/**
 * @brief Finds the greatest element in double precision CRS matrices
 *(max_element(C[0:nnz]))
 * @param C double precision monolish CRS matrix (size M x N)
 * @return greatest value
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 **/
double max(const matrix::CRS<double> &C);
/**
 * @brief Finds the greatest element in double precision CRS matrices
 *(max_element(C[0:nnz]))
 * @param C double precision monolish CRS matrix (size M x N)
 * @return greatest value
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 **/
double min(const matrix::CRS<double> &C);

//////////////////////////////////////////////////////
//  LinearOperator
//////////////////////////////////////////////////////

/**
 * @brief double precision element by element addition of LinearOperator A and
 * LinearOperator B.
 * @param A double precision monolish LinearOperator (size M x N)
 * @param B double precision monolish LinearOperator (size M x N)
 * @param C double precision monolish LinearOperator (size M x N)
 * @note
 * - # of computation: 2 functions
 * - Multi-threading: false
 * - GPU acceleration: false
 */
void add(const matrix::LinearOperator<double> &A,
         const matrix::LinearOperator<double> &B,
         matrix::LinearOperator<double> &C);

/**
 * @brief double precision element by element subtraction of LinearOperator A
 * and LinearOperator B.
 * @param A double precision monolish LinearOperator (size M x N)
 * @param B double precision monolish LinearOperator (size M x N)
 * @param C double precision monolish LinearOperator (size M x N)
 * @note
 * - # of computation: 2 functions
 * - Multi-threading: false
 * - GPU acceleration: false
 */
void sub(const matrix::LinearOperator<double> &A,
         const matrix::LinearOperator<double> &B,
         matrix::LinearOperator<double> &C);

/**
 * @brief double precision scalar and LinearOperator addition (C[i][j] = A[i][j]
 * + alpha)
 * @param A double precision monolish LinearOperator (size M x N)
 * @param alpha double precision scalar value
 * @param C double precision monolish LinearOperator (size M x N)
 * @note
 * - # of computation: 2 functions
 * - Multi-threading: false
 * - GPU acceleration: false
 */
void add(const matrix::LinearOperator<double> &A, const double &alpha,
         matrix::LinearOperator<double> &C);

/**
 * @brief double precision scalar and LinearOperator subtraction (C[i][j] =
 * A[i][j] - alpha)
 * @param A double precision monolish LinearOperator (size M x N)
 * @param alpha double precision scalar value
 * @param C double precision monolish LinearOperator (size M x N)
 * @note
 * - # of computation: 2 functions
 * - Multi-threading: false
 * - GPU acceleration: false
 */
void sub(const matrix::LinearOperator<double> &A, const double &alpha,
         matrix::LinearOperator<double> &C);

/**
 * @brief double precision scalar and LinearOperator multiplication (C[i][j] =
 * A[i][j] * alpha)
 * @param A double precision monolish LinearOperator (size M x N)
 * @param alpha double precision scalar value
 * @param C double precision monolish LinearOperator (size M x N)
 * @note
 * - # of computation: 2 functions
 * - Multi-threading: false
 * - GPU acceleration: false
 */
void mul(const matrix::LinearOperator<double> &A, const double &alpha,
         matrix::LinearOperator<double> &C);

/**
 * @brief double precision scalar and LinearOperator division (C[i][j] = A[i][j]
 * / alpha)
 * @param A double precision monolish LinearOperator (size M x N)
 * @param alpha double precision scalar value
 * @param C double precision monolish LinearOperator (size M x N)
 * @note
 * - # of computation: 2 functions
 * - Multi-threading: false
 * - GPU acceleration: false
 */
void div(const matrix::LinearOperator<double> &A, const double &alpha,
         matrix::LinearOperator<double> &C);

} // namespace vml
} // namespace monolish
