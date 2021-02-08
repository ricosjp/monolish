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
 * Basic Linear Algebra Subprograms for Dense Matrix, Sparse Matrix, Vector and
 * Scalar
 */
namespace blas {

//////////////////////////////////////////////////////
//  Copy
//////////////////////////////////////////////////////
/**
 * @brief single precision Dense matrix copy (y=a)
 * @param A single precision monolish Dense matrix (size M x N)
 * @param C single precision monolish Dense matrix (size M x N)
 * @note
 * - # of computation: M x N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
void copy(const matrix::Dense<float> &A, matrix::Dense<float> &C);

/**
 * @brief single precision Dense matrix copy (y=a)
 * @param A single precision monolish Dense matrix (size M x N)
 * @param C single precision monolish Dense matrix (size M x N)
 * @note
 * - # of computation: M x N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
void copy(const matrix::LinearOperator<float> &A,
          matrix::LinearOperator<float> &C);

/**
 * @brief single precision CRS matrix copy (y=a)
 * @param A single precision monolish CRS matrix (size M x N)
 * @param C single precision monolish CRS matrix (size M x N)
 * @note
 * - # of computation: M x N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 * @warning
 * A and C must be same non-zero structure
 */
void copy(const matrix::CRS<float> &A, matrix::CRS<float> &C);

//////////////////////////////////////////////////////
//  Matrix
//////////////////////////////////////////////////////

/**
 * @brief single precision Densematrix scal: A = alpha * A
 * @param alpha single precision scalar value
 * @param A single precision Dense matrix (size M x N)
 * @note
 * - # of computation: MN
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
void mscal(const float alpha, matrix::Dense<float> &A);

/**
 * @brief single precision CRS matrix scal: A = alpha * A
 * @param alpha single precision scalar value
 * @param A single precision CRS matrix (size M x N)
 * @note
 * - # of computation: MN
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
void mscal(const float alpha, matrix::CRS<float> &A);

///////////////

/**
 * @brief single precision CRS matrix addition: C = A + B (A and B must be
 * same non-zero structure)
 * @param A single precision CRS matrix (size M x N)
 * @param B single precision CRS matrix (size M x N)
 * @param C single precision CRS matrix (size M x N)
 * @note
 * - # of computation: nnz
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 * @warning
 * A and B must be same non-zero structure
 */
void matadd(const matrix::CRS<float> &A, const matrix::CRS<float> &B,
            matrix::CRS<float> &C);

/**
 * @brief single precision Dense matrix addition: C = A + B
 * @param A single precision Dense matrix (size M x N)
 * @param B single precision Dense matrix (size M x N)
 * @param C single precision Dense matrix (size M x N)
 * @note
 * - # of computation: MN
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
void matadd(const matrix::Dense<float> &A, const matrix::Dense<float> &B,
            matrix::Dense<float> &C);

/**
 * @brief single precision LinearOperator addition: C = A + B
 * @param A single precision LinearOperator (size M x N)
 * @param B single precision LinearOperator (size M x N)
 * @param C single precision LinearOperator (size M x N)
 * @note
 * - # of computation: 2 functions
 * - Multi-threading: false
 * - GPU acceleration: false
 */
void matadd(const matrix::LinearOperator<float> &A,
            const matrix::LinearOperator<float> &B,
            matrix::LinearOperator<float> &C);

/**
 * @brief single precision CRS matrix addition: C = A - B (A and B must be
 * same non-zero structure)
 * @param A single precision CRS matrix (size M x N)
 * @param B single precision CRS matrix (size M x N)
 * @param C single precision CRS matrix (size M x N)
 * @note
 * - # of computation: nnz
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 * @warning
 * A B must be same non-zero structure
 */
void matsub(const matrix::CRS<float> &A, const matrix::CRS<float> &B,
            matrix::CRS<float> &C);

/**
 * @brief single precision Dense matrix addition: C = A - B
 * @param A single precision Dense matrix (size M x N)
 * @param B single precision Dense matrix (size M x N)
 * @param C single precision Dense matrix (size M x N)
 * @note
 * - # of computation: MN
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
void matsub(const matrix::Dense<float> &A, const matrix::Dense<float> &B,
            matrix::Dense<float> &C);

/**
 * @brief single precision LinearOperator addition: C = A - B
 * @param A single precision LinearOperator (size M x N)
 * @param B single precision LinearOperator (size M x N)
 * @param C single precision LinearOperator (size M x N)
 * @note
 * - # of computation: 2 functions
 * - Multi-threading: false
 * - GPU acceleration: false
 */
void matsub(const matrix::LinearOperator<float> &A,
            const matrix::LinearOperator<float> &B,
            matrix::LinearOperator<float> &C);

///////////////

/**
 * @brief single precision Dense matrix and vector multiplication: y = Ax
 * @param A single precision Dense matrix (size M x N)
 * @param x single precision monolish vector (size M)
 * @param y single precision monolish vector (size M)
 * @note
 * - # of computation: MN
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
void matvec(const matrix::Dense<float> &A, const vector<float> &x,
            vector<float> &y);

/**
 * @brief single precision sparse matrix (CRS) and vector multiplication: y = Ax
 * @param A single precision CRS matrix (size M x N)
 * @param x single precision monolish vector (size M)
 * @param y single precision monolish vector (size M)
 * @note
 * - # of computation: 2nnz
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
void matvec(const matrix::CRS<float> &A, const vector<float> &x,
            vector<float> &y);

/**
 * @brief single precision matrix (LinearOperator) and vector multiplication: y
 * = Ax
 * @param A single precision LinearOperator (size M x N)
 * @param x single precision monolish vector (size N)
 * @param y single precision monolish vector (size M)
 * @note
 * - # of computation: depends on matvec function
 * - Multi-threading: depends on matvec function
 * - GPU acceleration: depends on matvec function
 */
void matvec(const matrix::LinearOperator<float> &A, const vector<float> &x,
            vector<float> &y);

/**
 * @brief single precision (Hermitian) transposed matrix (LinearOperator) and
 * vector multiplication: y = A^T x
 * @param A single precision LinearOperator (size M x N)
 * @param x single precision monolish vector (size M)
 * @param y single precision monolish vector (size N)
 * @note
 * - # of computation: depends on matvec function
 * - Multi-threading: depends on matvec function
 * - GPU acceleration: depends on matvec function
 */
void rmatvec(const matrix::LinearOperator<float> &A, const vector<float> &x,
             vector<float> &y);

///////////////

/**
 * @brief single precision Dense matrix multiplication: C = AB
 * @param A single precision Dense matrix (size M x K)
 * @param B single precision Dense matrix (size K x N)
 * @param C single precision Dense matrix (size M x N)
 * @note
 * - # of computation: 2MNK
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
void matmul(const matrix::Dense<float> &A, const matrix::Dense<float> &B,
            matrix::Dense<float> &C);

/**
 * @brief single precision Dense matrix multiplication: C = AB
 * @param A single precision CRS matrix (size M x K)
 * @param B single precision Dense matrix (size K x N)
 * @param C single precision Dense matrix (size M x N)
 * @note
 * - # of computation: 2*N*nnz
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
void matmul(const matrix::CRS<float> &A, const matrix::Dense<float> &B,
            matrix::Dense<float> &C);

/**
 * @brief single precision LinearOperator multiplication: C = AB
 * @param A single precision LinearOperator (size M x K)
 * @param B single precision LinearOperator (size K x N)
 * @param C single precision LinearOperator (size M x N)
 * @note
 * - # of computation: 2 functions
 * - Multi-threading: false
 * - GPU acceleration: false
 */
void matmul(const matrix::LinearOperator<float> &A,
            const matrix::LinearOperator<float> &B,
            matrix::LinearOperator<float> &C);

} // namespace blas
} // namespace monolish
