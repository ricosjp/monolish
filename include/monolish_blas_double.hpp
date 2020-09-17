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
 * @param[in] x double precision monolish vector
 * @return The result of the asum
 * @note 
 * - Order: N
 */
double asum(const vector<double> &x);

/**
 * @brief double precision vector asum (absolute sum)
 * @param[in] x double precision monolish vector
 * @param[in] ans result value
 * @note 
 * - Order: N
 * - memory allocation: none
 */
void asum(const vector<double> &x, double &ans);

/**
 * @brief double precision vector sum
 * @param[in] x double precision monolish vector
 * @return The result of the sum
 */
double sum(const vector<double> &x);

/**
 * @brief double precision vector sum
 * @param[in] x double precision monolish vector
 * @param[in] ans result value
 */
void sum(const vector<double> &x, double &ans);

/**
 * @brief double precision axpy: y = ax + y
 * @param[in] alpha double precision scalar value
 * @param[in] x double precision monolish vector
 * @param[in] y double precision monolish vector
 */
void axpy(const double alpha, const vector<double> &x, vector<double> &y);

/**
 * @brief double precision axpyz: z = ax + y
 * @param[in] alpha double precision scalar value
 * @param[in] x double precision monolish vector
 * @param[in] y double precision monolish vector
 * @param[in] z double precision monolish vector
 */
void axpyz(const double alpha, const vector<double> &x, const vector<double> &y,
           vector<double> &z);

/**
 * @brief double precision inner product (dot)
 * @param[in] x double precision monolish vector
 * @param[in] y double precision monolish vector
 * @return The result of the inner product product of x and y
 */
double dot(const vector<double> &x, const vector<double> &y);

/**
 * @brief double precision inner product (dot)
 * @param[in] x double precision monolish vector
 * @param[in] y double precision monolish vector
 * @param[in] ans result value
 */
void dot(const vector<double> &x, const vector<double> &y, double &ans);

/**
 * @brief double precision nrm2: ||x||_2
 * @param[in] x double precision monolish vector
 * @return The result of the nrm2
 */
double nrm2(const vector<double> &x);

/**
 * @brief double precision nrm2: ||x||_2
 * @param[in] x double precision monolish vector
 * @param[in] ans result value
 */
void nrm2(const vector<double> &x, double &ans);

/**
 * @brief double precision scal: x = alpha * x
 * @param[in] alpha double precision scalar value
 * @param[in] x double precision monolish vector
 */
void scal(const double alpha, vector<double> &x);

/**
 * @brief double precision xpay: y = x + ay
 * @param[in] alpha double precision scalar value
 * @param[in] x double precision monolish vector
 * @param[in] y double precision monolish vector
 * @param[in] z double precision monolish vector
 */
void xpay(const double alpha, const vector<double> &x, vector<double> &y);

//////////////////////////////////////////////////////
//  Matrix
//////////////////////////////////////////////////////

/**
 * @brief double precision scal: A = alpha * A
 * @param[in] alpha double precision scalar value
 * @param[in] A double precision CRS matrix
 */
void mscal(const double alpha, matrix::Dense<double> &A);

/**
 * @brief double precision scal: A = alpha * A
 * @param[in] alpha double precision scalar value
 * @param[in] A double precision CRS matrix
 */
void mscal(const double alpha, matrix::CRS<double> &A);

/**
 * @brief double precision Dense matrix and vector multiplication: y = Ax
 * @param[in] A double precision Dense matrix
 * @param[in] x double precision monolish vector
 * @param[in] y double precision monolish vector
 */
void matvec(const matrix::Dense<double> &A, const vector<double> &x,
            vector<double> &y);

/**
 * @brief double precision sparse matrix (CRS) and vector multiplication: y = Ax
 * @param[in] A double precision CRS matrix
 * @param[in] x double precision monolish vector
 * @param[in] y double precision monolish vector
 */
void matvec(const matrix::CRS<double> &A, const vector<double> &x,
            vector<double> &y);

/**
 * @brief double precision Dense matrix addition: C = AB (A and B must be same
 * structure)
 * @param[in] A double precision CRS matrix
 * @param[in] B double precision CRS matrix
 * @param[in] C double precision CRS matrix
 */
void matadd(const matrix::CRS<double> &A, const matrix::CRS<double> &B,
            matrix::CRS<double> &C);

/**
 * @brief double precision Dense matrix addition: C = AB
 * @param[in] A double precision Dense matrix
 * @param[in] B double precision Dense matrix
 * @param[in] C double precision Dense matrix
 */
void matadd(const matrix::Dense<double> &A, const matrix::Dense<double> &B,
            matrix::Dense<double> &C);

/**
 * @brief double precision Dense matrix multiplication: C = AB
 * @param[in] A double precision CRS matrix
 * @param[in] B double precision Dense matrix
 * @param[in] C double precision Dense matrix
 */
void matmul(const matrix::CRS<double> &A, const matrix::Dense<double> &B,
            matrix::Dense<double> &C);

/**
 * @brief double precision Dense matrix multiplication: C = AB
 * @param[in] A double precision Dense matrix
 * @param[in] B double precision Dense matrix
 * @param[in] C double precision Dense matrix
 */
void matmul(const matrix::Dense<double> &A, const matrix::Dense<double> &B,
            matrix::Dense<double> &C);

} // namespace blas
} // namespace monolish
