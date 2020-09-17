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
 * @brief float precision vector asum (absolute sum)
 * @param[in] x float precision monolish vector
 * @return The result of the asum
 * @note 
 * - Order: N
 */
float asum(const vector<float> &x);

/**
 * @brief float precision vector asum (absolute sum)
 * @param[in] x float precision monolish vector
 * @param[in] ans result value
 * @note 
 * - Order: N
 * - memory allocation: none
 */
 */
void asum(const vector<float> &x, float &ans);

/**
 * @brief float precision vector sum
 * @param[in] x float precision monolish vector
 * @return The result of the sum
 */
float sum(const vector<float> &x);

/**
 * @brief float precision vector sum
 * @param[in] x float precision monolish vector
 * @param[in] ans result value
 */
void sum(const vector<float> &x, float &ans);

/**
 * @brief float precision axpy: y = ax + y
 * @param[in] alpha float precision scalar value
 * @param[in] x float precision monolish vector
 * @param[in] y float precision monolish vector
 */
void axpy(const float alpha, const vector<float> &x, vector<float> &y);

/**
 * @brief float precision axpyz: z = ax + y
 * @param[in] alpha float precision scalar value
 * @param[in] x float precision monolish vector
 * @param[in] y float precision monolish vector
 * @param[in] z float precision monolish vector
 */
void axpyz(const float alpha, const vector<float> &x, const vector<float> &y,
           vector<float> &z);

/**
 * @brief float precision inner product (dot)
 * @param[in] x float precision monolish vector
 * @param[in] y float precision monolish vector
 * @return The result of the inner product product of x and y
 */
float dot(const vector<float> &x, const vector<float> &y);

/**
 * @brief float precision inner product (dot)
 * @param[in] x float precision monolish vector
 * @param[in] y float precision monolish vector
 * @param[in] ans result value
 */
void dot(const vector<float> &x, const vector<float> &y, float &ans);

/**
 * @brief float precision nrm2: ||x||_2
 * @param[in] x float precision monolish vector
 * @return The result of the nrm2
 */
float nrm2(const vector<float> &x);

/**
 * @brief float precision nrm2: ||x||_2
 * @param[in] x float precision monolish vector
 * @param[in] ans result value
 */
void nrm2(const vector<float> &x, float &ans);

/**
 * @brief float precision scal: x = alpha * x
 * @param[in] alpha float precision scalar value
 * @param[in] x float precision monolish vector
 */
void scal(const float alpha, vector<float> &x);

/**
 * @brief float precision xpay: y = x + ay
 * @param[in] alpha float precision scalar value
 * @param[in] x float precision monolish vector
 * @param[in] y float precision monolish vector
 * @param[in] z float precision monolish vector
 */
void xpay(const float alpha, const vector<float> &x, vector<float> &y);

//////////////////////////////////////////////////////
//  Matrix
//////////////////////////////////////////////////////

/**
 * @brief float precision scal: A = alpha * A
 * @param[in] alpha float precision scalar value
 * @param[in] A float precision CRS matrix
 */
void mscal(const float alpha, matrix::Dense<float> &A);

/**
 * @brief float precision scal: A = alpha * A
 * @param[in] alpha float precision scalar value
 * @param[in] A float precision CRS matrix
 */
void mscal(const float alpha, matrix::CRS<float> &A);

/**
 * @brief float precision Dense matrix and vector multiplication: y = Ax
 * @param[in] A float precision Dense matrix
 * @param[in] x float precision monolish vector
 * @param[in] y float precision monolish vector
 */
void matvec(const matrix::Dense<float> &A, const vector<float> &x,
            vector<float> &y);

/**
 * @brief float precision sparse matrix (CRS) and vector multiplication: y = Ax
 * @param[in] A float precision CRS matrix
 * @param[in] x float precision monolish vector
 * @param[in] y float precision monolish vector
 */
void matvec(const matrix::CRS<float> &A, const vector<float> &x,
            vector<float> &y);

/**
 * @brief float precision Dense matrix addition: C = AB (A and B must be same
 * structure)
 * @param[in] A float precision CRS matrix
 * @param[in] B float precision CRS matrix
 * @param[in] C float precision CRS matrix
 */
void matadd(const matrix::CRS<float> &A, const matrix::CRS<float> &B,
            matrix::CRS<float> &C);

/**
 * @brief float precision Dense matrix addition: C = AB
 * @param[in] A float precision Dense matrix
 * @param[in] B float precision Dense matrix
 * @param[in] C float precision Dense matrix
 */
void matadd(const matrix::Dense<float> &A, const matrix::Dense<float> &B,
            matrix::Dense<float> &C);

/**
 * @brief float precision Dense matrix multiplication: C = AB
 * @param[in] A float precision CRS matrix
 * @param[in] B float precision Dense matrix
 * @param[in] C float precision Dense matrix
 */
void matmul(const matrix::CRS<float> &A, const matrix::Dense<float> &B,
            matrix::Dense<float> &C);

/**
 * @brief float precision Dense matrix multiplication: C = AB
 * @param[in] A float precision Dense matrix
 * @param[in] B float precision Dense matrix
 * @param[in] C float precision Dense matrix
 */
void matmul(const matrix::Dense<float> &A, const matrix::Dense<float> &B,
            matrix::Dense<float> &C);

} // namespace blas
} // namespace monolish
