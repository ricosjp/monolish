#pragma once
#include "common/monolish_common.hpp"
#include <stdio.h>
#include <vector>

#if defined USE_MPI
#include <mpi.h>
#endif

namespace monolish {
namespace blas {

//////////////////////////////////////////////////////
//  Vector
//////////////////////////////////////////////////////
/**
 * @brief double precision vector asum (absolute sum)
 * @param x double precision monolish vector (size N)
 * @return The result of the asum
 * @note
 * - # of computation: N
 * - Multi-threading (OpenMP): true
 * - GPU acceleration (OpenACC): true
 */
double asum(const vector<double> &x);

/**
 * @brief double precision vector asum (absolute sum)
 * @param x double precision monolish vector (size N)
 * @param ans The result of the asum
 * @note
 * - # of computation: N
 * - Multi-threading (OpenMP): true
 * - GPU acceleration (OpenACC): true
 */
void asum(const vector<double> &x, double &ans);

/**
 * @brief double precision vector sum
 * @param x double precision monolish vector (size N)
 * @return The result of the sum
 * @note
 * - # of computation: N
 * - Multi-threading (OpenMP): true
 * - GPU acceleration (OpenACC): true
 */
double sum(const vector<double> &x);

/**
 * @brief double precision vector sum
 * @param x double precision monolish vector (size N)
 * @param ans The result of the sum
 * @note
 * - # of computation: N
 * - Multi-threading (OpenMP): true
 * - GPU acceleration (OpenACC): true
 */
void sum(const vector<double> &x, double &ans);

/**
 * @brief double precision axpy: y = ax + y
 * @param alpha double precision scalar value
 * @param x double precision monolish vector (size N)
 * @param y double precision monolish vector (size N)
 * @note
 * - # of computation: N
 * - Multi-threading (OpenMP): true
 * - GPU acceleration (OpenACC): true
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
 * - Multi-threading (OpenMP): true
 * - GPU acceleration (OpenACC): true
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
 * - Multi-threading (OpenMP): true
 * - GPU acceleration (OpenACC): true
 */
double dot(const vector<double> &x, const vector<double> &y);

/**
 * @brief double precision inner product (dot)
 * @param x double precision monolish vector (size N)
 * @param y double precision monolish vector (size N)
 * @param ans The result of the inner product product of x and y
 * @note
 * - # of computation: 2N
 * - Multi-threading (OpenMP): true
 * - GPU acceleration (OpenACC): true
 */
void dot(const vector<double> &x, const vector<double> &y, double &ans);

/**
 * @brief double precision nrm2: ||x||_2
 * @param x double precision monolish vector (size N)
 * @return The result of the nrm2
 * @note
 * - # of computation: 2N
 * - Multi-threading (OpenMP): true
 * - GPU acceleration (OpenACC): true
 */
double nrm2(const vector<double> &x);

/**
 * @brief double precision nrm2: ||x||_2
 * @param x double precision monolish vector (size N)
 * @param ans The result of the nrm2
 * @note
 * - # of computation: 2N
 * - Multi-threading (OpenMP): true
 * - GPU acceleration (OpenACC): true
 */
void nrm2(const vector<double> &x, double &ans);

/**
 * @brief double precision scal: x = alpha * x
 * @param alpha double precision scalar value
 * @param x double precision monolish vector (size N)
 * @note
 * - # of computation: N
 * - Multi-threading (OpenMP): true
 * - GPU acceleration (OpenACC): true
 */
void scal(const double alpha, vector<double> &x);

/**
 * @brief double precision xpay: y = x + ay
 * @param alpha double precision scalar value
 * @param x double precision monolish vector (size N)
 * @param y double precision monolish vector (size N)
 * @note
 * - # of computation: 2N
 * - Multi-threading (OpenMP): true
 * - GPU acceleration (OpenACC): true
 */
void xpay(const double alpha, const vector<double> &x, vector<double> &y);

//////////////////////////////////////////////////////
//  Matrix
//////////////////////////////////////////////////////

/**
 * @brief double precision scal: A = alpha * A
 * @param alpha double precision scalar value
 * @param A double precision Dense matrix (size M x N)
 * @note
 * - # of computation: MN
 * - Multi-threading (OpenMP): true
 * - GPU acceleration (OpenACC): true
 */
void mscal(const double alpha, matrix::Dense<double> &A);

/**
 * @brief double precision scal: A = alpha * A
 * @param alpha double precision scalar value
 * @param A double precision CRS matrix (size M x N)
 * @note
 * - # of computation: MN
 * - Multi-threading (OpenMP): true
 * - GPU acceleration (OpenACC): true
 */
void mscal(const double alpha, matrix::CRS<double> &A);

///////////////

/**
 * @brief double precision Dense matrix addition: C = A + B (A and B must be
 * same non-zero structure)
 * @param A double precision CRS matrix (size M x N)
 * @param B double precision CRS matrix (size M x N)
 * @param C double precision CRS matrix (size M x N)
 * @note
 * - # of computation: nnz
 * - Multi-threading (OpenMP): true
 * - GPU acceleration (OpenACC): true
 * @warning
 * A and B must be same non-zero structure
 */
void matadd(const matrix::CRS<double> &A, const matrix::CRS<double> &B,
            matrix::CRS<double> &C);

/**
 * @brief double precision Dense matrix addition: C = AB
 * @param A double precision Dense matrix (size M x N)
 * @param B double precision Dense matrix (size M x N)
 * @param C double precision Dense matrix (size M x N)
 * @note
 * - # of computation: MN
 * - Multi-threading (OpenMP): true
 * - GPU acceleration (OpenACC): true
 */
void matadd(const matrix::Dense<double> &A, const matrix::Dense<double> &B,
            matrix::Dense<double> &C);

///////////////

/**
 * @brief double precision Dense matrix and vector multiplication: y = Ax
 * @param A double precision Dense matrix (size M x N)
 * @param x double precision monolish vector (size N)
 * @param y double precision monolish vector (size N)
 * @note
 * - # of computation: MN
 * - Multi-threading (OpenMP): true
 * - GPU acceleration (OpenACC): true
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
 * - Multi-threading (OpenMP): true
 * - GPU acceleration (OpenACC): true
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
 * - Multi-threading (OpenMP): true
 * - GPU acceleration (OpenACC): true
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
 * - Multi-threading (OpenMP): true
 * - GPU acceleration (OpenACC): true
 */
void matmul(const matrix::CRS<double> &A, const matrix::Dense<double> &B,
            matrix::Dense<double> &C);

} // namespace blas
} // namespace monolish
