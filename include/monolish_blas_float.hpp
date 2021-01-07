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
//  Vector
//////////////////////////////////////////////////////
/**
 * @brief single precision element by element addition of vector a and vector b.
 * @param a single precision monolish vector (size N)
 * @param b single precision monolish vector (size N)
 * @param y single precision monolish vector (size N)
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
void vecadd(const vector<float> &a, const vector<float> &b, vector<float> &y);

/**
 * @brief single precision element by element subtraction of vector a and vector
 * b.
 * @param a single precision monolish vector (size N)
 * @param b single precision monolish vector (size N)
 * @param y single precision monolish vector (size N)
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
void vecsub(const vector<float> &a, const vector<float> &b, vector<float> &y);
/**
 * @brief single precision vector asum (absolute sum)
 * @param x single precision monolish vector (size N)
 * @return The result of the asum
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
float asum(const vector<float> &x);

/**
 * @brief single precision vector asum (absolute sum)
 * @param x single precision monolish vector (size N)
 * @param ans The result of the asum
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
void asum(const vector<float> &x, float &ans);

/**
 * @brief single precision vector sum
 * @param x single precision monolish vector (size N)
 * @return The result of the sum
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
float sum(const vector<float> &x);

/**
 * @brief single precision vector sum
 * @param x single precision monolish vector (size N)
 * @param ans The result of the sum
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
void sum(const vector<float> &x, float &ans);

/**
 * @brief single precision axpy: y = ax + y
 * @param alpha single precision scalar value
 * @param x single precision monolish vector (size N)
 * @param y single precision monolish vector (size N)
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
void axpy(const float alpha, const vector<float> &x, vector<float> &y);

/**
 * @brief single precision axpyz: z = ax + y
 * @param alpha single precision scalar value
 * @param x single precision monolish vector (size N)
 * @param y single precision monolish vector (size N)
 * @param z single precision monolish vector (size N)
 * @note
 * - # of computation: 2N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
void axpyz(const float alpha, const vector<float> &x, const vector<float> &y,
           vector<float> &z);

/**
 * @brief single precision inner product (dot)
 * @param x single precision monolish vector (size N)
 * @param y single precision monolish vector (size N)
 * @return The result of the inner product product of x and y
 * @note
 * - # of computation: 2N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
float dot(const vector<float> &x, const vector<float> &y);

/**
 * @brief single precision inner product (dot)
 * @param x single precision monolish vector (size N)
 * @param y single precision monolish vector (size N)
 * @param ans The result of the inner product product of x and y
 * @note
 * - # of computation: 2N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
void dot(const vector<float> &x, const vector<float> &y, float &ans);

/**
 * @brief single precision nrm2: ||x||_2
 * @param x single precision monolish vector (size N)
 * @return The result of the nrm2
 * @note
 * - # of computation: 2N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
float nrm2(const vector<float> &x);

/**
 * @brief single precision nrm2: ||x||_2
 * @param x single precision monolish vector (size N)
 * @param ans The result of the nrm2
 * @note
 * - # of computation: 2N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
void nrm2(const vector<float> &x, float &ans);

/**
 * @brief single precision scal: x = alpha * x
 * @param alpha single precision scalar value
 * @param x single precision monolish vector (size N)
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
void scal(const float alpha, vector<float> &x);

/**
 * @brief single precision xpay: y = x + ay
 * @param alpha single precision scalar value
 * @param x single precision monolish vector (size N)
 * @param y single precision monolish vector (size N)
 * @note
 * - # of computation: 2N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
void xpay(const float alpha, const vector<float> &x, vector<float> &y);

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
 * A and B must be same non-zero structure
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

///////////////

/**
 * @brief single precision Dense matrix and vector multiplication: y = Ax
 * @param A single precision Dense matrix (size M x N)
 * @param x single precision monolish vector (size N)
 * @param y single precision monolish vector (size N)
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
 * @param x single precision monolish vector (size N)
 * @param y single precision monolish vector (size N)
 * @note
 * - # of computation: 2nnz
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
void matvec(const matrix::CRS<float> &A, const vector<float> &x,
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

} // namespace blas
} // namespace monolish
