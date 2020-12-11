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

void add(const vector<double> &a, const vector<double> &b, vector<double> &y);
void add(const vector<double> &a, const double alpha, vector<double> &y);

void sub(const vector<double> &a, const vector<double> &b, vector<double> &y);
void sub(const vector<double> &a, const double alpha, vector<double> &y);

void mul(const vector<double> &a, const vector<double> &b, vector<double> &y);
void mul(const vector<double> &a, const double alpha, vector<double> &y);

void div(const vector<double> &a, const vector<double> &b, vector<double> &y);
void div(const vector<double> &a, const double alpha, vector<double> &y);

void vecadd(const vector<double> &a, const vector<double> &b, vector<double> &y);
void vecsub(const vector<double> &a, const vector<double> &b, vector<double> &y);
//////////////////////////////////////////////////////
//  Vector
//////////////////////////////////////////////////////
/**
 * @brief double precision vector asum (absolute sum)
 * @param x double precision monolish vector (size N)
 * @return The result of the asum
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
double asum(const vector<double> &x);

/**
 * @brief double precision vector asum (absolute sum)
 * @param x double precision monolish vector (size N)
 * @param ans The result of the asum
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
void asum(const vector<double> &x, double &ans);

/**
 * @brief double precision vector sum
 * @param x double precision monolish vector (size N)
 * @return The result of the sum
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
double sum(const vector<double> &x);

/**
 * @brief double precision vector sum
 * @param x double precision monolish vector (size N)
 * @param ans The result of the sum
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
void sum(const vector<double> &x, double &ans);

/**
 * @brief double precision axpy: y = ax + y
 * @param alpha double precision scalar value
 * @param x double precision monolish vector (size N)
 * @param y double precision monolish vector (size N)
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
void axpy(const double alpha, const vector<double> &x, vector<double> &y);

/**
 * @brief double precision axpyz: z = ax + y
 * @param alpha double precision scalar value
 * @param x double precision monolish vector (size N)
 * @param y double precision monolish vector (size N)
 * @param z double precision monolish vector (size N)
 * @note
 * - # of computation: 2N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
void axpyz(const double alpha, const vector<double> &x, const vector<double> &y,
           vector<double> &z);

/**
 * @brief double precision inner product (dot)
 * @param x double precision monolish vector (size N)
 * @param y double precision monolish vector (size N)
 * @return The result of the inner product product of x and y
 * @note
 * - # of computation: 2N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
double dot(const vector<double> &x, const vector<double> &y);

/**
 * @brief double precision inner product (dot)
 * @param x double precision monolish vector (size N)
 * @param y double precision monolish vector (size N)
 * @param ans The result of the inner product product of x and y
 * @note
 * - # of computation: 2N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
void dot(const vector<double> &x, const vector<double> &y, double &ans);

/**
 * @brief double precision nrm2: ||x||_2
 * @param x double precision monolish vector (size N)
 * @return The result of the nrm2
 * @note
 * - # of computation: 2N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
double nrm2(const vector<double> &x);

/**
 * @brief double precision nrm2: ||x||_2
 * @param x double precision monolish vector (size N)
 * @param ans The result of the nrm2
 * @note
 * - # of computation: 2N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
void nrm2(const vector<double> &x, double &ans);

/**
 * @brief double precision scal: x = alpha * x
 * @param alpha double precision scalar value
 * @param x double precision monolish vector (size N)
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
void scal(const double alpha, vector<double> &x);

/**
 * @brief double precision xpay: y = x + ay
 * @param alpha double precision scalar value
 * @param x double precision monolish vector (size N)
 * @param y double precision monolish vector (size N)
 * @note
 * - # of computation: 2N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
void xpay(const double alpha, const vector<double> &x, vector<double> &y);

//////////////////////////////////////////////////////
//  Matrix
//////////////////////////////////////////////////////

/**
 * @brief double precision Densematrix scal: A = alpha * A
 * @param alpha double precision scalar value
 * @param A double precision Dense matrix (size M x N)
 * @note
 * - # of computation: MN
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
void mscal(const double alpha, matrix::Dense<double> &A);

/**
 * @brief double precision CRS matrix scal: A = alpha * A
 * @param alpha double precision scalar value
 * @param A double precision CRS matrix (size M x N)
 * @note
 * - # of computation: MN
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
void mscal(const double alpha, matrix::CRS<double> &A);

///////////////

/**
 * @brief double precision CRS matrix addition: C = A + B (A and B must be
 * same non-zero structure)
 * @param A double precision CRS matrix (size M x N)
 * @param B double precision CRS matrix (size M x N)
 * @param C double precision CRS matrix (size M x N)
 * @note
 * - # of computation: nnz
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 * @warning
 * A and B must be same non-zero structure
 */
void matadd(const matrix::CRS<double> &A, const matrix::CRS<double> &B,
            matrix::CRS<double> &C);

/**
 * @brief double precision Dense matrix addition: C = A + B
 * @param A double precision Dense matrix (size M x N)
 * @param B double precision Dense matrix (size M x N)
 * @param C double precision Dense matrix (size M x N)
 * @note
 * - # of computation: MN
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
void matadd(const matrix::Dense<double> &A, const matrix::Dense<double> &B,
            matrix::Dense<double> &C);

/**
 * @brief double precision CRS matrix addition: C = A - B (A and B must be
 * same non-zero structure)
 * @param A double precision CRS matrix (size M x N)
 * @param B double precision CRS matrix (size M x N)
 * @param C double precision CRS matrix (size M x N)
 * @note
 * - # of computation: nnz
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 * @warning
 * A and B must be same non-zero structure
 */
void matsub(const matrix::CRS<double> &A, const matrix::CRS<double> &B,
            matrix::CRS<double> &C);

/**
 * @brief double precision Dense matrix addition: C = A - B
 * @param A double precision Dense matrix (size M x N)
 * @param B double precision Dense matrix (size M x N)
 * @param C double precision Dense matrix (size M x N)
 * @note
 * - # of computation: MN
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
void matsub(const matrix::Dense<double> &A, const matrix::Dense<double> &B,
            matrix::Dense<double> &C);

///////////////

/**
 * @brief double precision Dense matrix and vector multiplication: y = Ax
 * @param A double precision Dense matrix (size M x N)
 * @param x double precision monolish vector (size N)
 * @param y double precision monolish vector (size N)
 * @note
 * - # of computation: MN
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
void matvec(const matrix::Dense<double> &A, const vector<double> &x,
            vector<double> &y);

/**
 * @brief double precision sparse matrix (CRS) and vector multiplication: y = Ax
 * @param A double precision CRS matrix (size M x N)
 * @param x double precision monolish vector (size N)
 * @param y double precision monolish vector (size N)
 * @note
 * - # of computation: 2nnz
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
void matvec(const matrix::CRS<double> &A, const vector<double> &x,
            vector<double> &y);

///////////////

/**
 * @brief double precision Dense matrix multiplication: C = AB
 * @param A double precision Dense matrix (size M x K)
 * @param B double precision Dense matrix (size K x N)
 * @param C double precision Dense matrix (size M x N)
 * @note
 * - # of computation: 2MNK
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
void matmul(const matrix::Dense<double> &A, const matrix::Dense<double> &B,
            matrix::Dense<double> &C);

/**
 * @brief double precision Dense matrix multiplication: C = AB
 * @param A double precision CRS matrix (size M x K)
 * @param B double precision Dense matrix (size K x N)
 * @param C double precision Dense matrix (size M x N)
 * @note
 * - # of computation: 2*N*nnz
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
void matmul(const matrix::CRS<double> &A, const matrix::Dense<double> &B,
            matrix::Dense<double> &C);

} // namespace blas
} // namespace monolish
