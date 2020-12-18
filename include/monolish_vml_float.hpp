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
namespace vml {

/**
 * @brief tanh to single precision vector elements (y[0:N] = tanh(a[0:N]))
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 **/
void tanh(const vector<float> &a, vector<float> &y);

void tanh(const matrix::Dense<float> &A, matrix::Dense<float> &C);
void tanh(const matrix::CRS<float> &A, matrix::CRS<float> &C);

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
void add(const vector<float> &a, const vector<float> &b, vector<float> &y);

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
void sub(const vector<float> &a, const vector<float> &b, vector<float> &y);

/**
 * @brief single precision element by element multiplication of vector a and
 * vector b.
 * @param a single precision monolish vector (size N)
 * @param b single precision monolish vector (size N)
 * @param y single precision monolish vector (size N)
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
void mul(const vector<float> &a, const vector<float> &b, vector<float> &y);

/**
 * @brief single precision element by element division of vector a and vector b.
 * @param a single precision monolish vector (size N)
 * @param b single precision monolish vector (size N)
 * @param y single precision monolish vector (size N)
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
void div(const vector<float> &a, const vector<float> &b, vector<float> &y);

/**
 * @brief single precision scalar and vector addition (y[i] = a[i] + alpha)
 * @param a single precision monolish vector (size N)
 * @param alpha single precision scalar value
 * @param y single precision monolish vector (size N)
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
void add(const vector<float> &a, const float alpha, vector<float> &y);

/**
 * @brief single precision scalar and vector subtraction (y[i] = a[i] - alpha)
 * @param a single precision monolish vector (size N)
 * @param alpha single precision scalar value
 * @param y single precision monolish vector (size N)
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
void sub(const vector<float> &a, const float alpha, vector<float> &y);

/**
 * @brief single precision scalar and vector multiplication (y[i] = a[i] *
 * alpha)
 * @param a single precision monolish vector (size N)
 * @param alpha single precision scalar value
 * @param y single precision monolish vector (size N)
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
void mul(const vector<float> &a, const float alpha, vector<float> &y);

/**
 * @brief single precision scalar and vector division (y[i] = a[i] / alpha)
 * @param a single precision monolish vector (size N)
 * @param alpha single precision scalar value
 * @param y single precision monolish vector (size N)
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
void div(const vector<float> &a, const float alpha, vector<float> &y);

//////////////////////////////////////////////////////
// Dense
//////////////////////////////////////////////////////

/**
 * @brief single precision element by element addition of Dense matrix A and
 * Dense matrix B.
 * @param A single precision monolish Dense Matrix (size M x N)
 * @param B single precision monolish Dense Matrix (size M x N)
 * @param C single precision monolish Dense Matrix (size M x N)
 * @note
 * - # of computation: MN
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
void add(const matrix::Dense<float> &A, const matrix::Dense<float> &B,
         matrix::Dense<float> &C);

/**
 * @brief single precision element by element subtraction of Dense matrix A and
 * Dense matrix B.
 * @param A single precision monolish Dense Matrix (size M x N)
 * @param B single precision monolish Dense Matrix (size M x N)
 * @param C single precision monolish Dense Matrix (size M x N)
 * @note
 * - # of computation: MN
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
void sub(const matrix::Dense<float> &A, const matrix::Dense<float> &B,
         matrix::Dense<float> &C);

/**
 * @brief single precision element by element multiplication of Dense matrix A
 * and Dense matrix B.
 * @param A single precision monolish Dense Matrix (size M x N)
 * @param B single precision monolish Dense Matrix (size M x N)
 * @param C single precision monolish Dense Matrix (size M x N)
 * @note
 * - # of computation: MN
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
void mul(const matrix::Dense<float> &A, const matrix::Dense<float> &B,
         matrix::Dense<float> &C);

/**
 * @brief single precision element by element division of Dense matrix A and
 * Dense matrix B.
 * @param A single precision monolish Dense Matrix (size M x N)
 * @param B single precision monolish Dense Matrix (size M x N)
 * @param C single precision monolish Dense Matrix (size M x N)
 * @note
 * - # of computation: MN
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
void div(const matrix::Dense<float> &A, const matrix::Dense<float> &B,
         matrix::Dense<float> &C);

/**
 * @brief single precision scalar and Dence matrix addition (C[i][j] = A[i][j] +
 * alpha)
 * @param A single precision monolish Dense Matrix (size M x N)
 * @param alpha single precision scalar value
 * @param C single precision monolish Dense Matrix (size M x N)
 * @note
 * - # of computation: MN
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
void add(const matrix::Dense<float> &A, const float alpha,
         matrix::Dense<float> &C);

/**
 * @brief single precision scalar and Dence matrix subtraction (C[i][j] =
 * A[i][j] + alpha)
 * @param A single precision monolish Dense Matrix (size M x N)
 * @param alpha single precision scalar value
 * @param C single precision monolish Dense Matrix (size M x N)
 * @note
 * - # of computation: MN
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
void sub(const matrix::Dense<float> &A, const float alpha,
         matrix::Dense<float> &C);

/**
 * @brief single precision scalar and Dence matrix multiplication (C[i][j] =
 * A[i][j] + alpha)
 * @param A single precision monolish Dense Matrix (size M x N)
 * @param alpha single precision scalar value
 * @param C single precision monolish Dense Matrix (size M x N)
 * @note
 * - # of computation: MN
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
void mul(const matrix::Dense<float> &A, const float alpha,
         matrix::Dense<float> &C);

/**
 * @brief single precision scalar and Dence matrix division (C[i][j] = A[i][j] +
 * alpha)
 * @param A single precision monolish Dense Matrix (size M x N)
 * @param alpha single precision scalar value
 * @param C single precision monolish Dense Matrix (size M x N)
 * @note
 * - # of computation: MN
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
void div(const matrix::Dense<float> &A, const float alpha,
         matrix::Dense<float> &C);

//////////////////////////////////////////////////////
// CRS
//////////////////////////////////////////////////////

/**
 * @brief single precision element by element addition of CRS matrix A and CRS
 * matrix B.
 * @param A single precision monolish CRS Matrix (size M x N)
 * @param B single precision monolish CRS Matrix (size M x N)
 * @param C single precision monolish CRS Matrix (size M x N)
 * @note
 * - # of computation: MN
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
void add(const matrix::CRS<float> &A, const matrix::CRS<float> &B,
         matrix::CRS<float> &C);

/**
 * @brief single precision element by element subtraction of CRS matrix A and
 * CRS matrix B.
 * @param A single precision monolish CRS Matrix (size M x N)
 * @param B single precision monolish CRS Matrix (size M x N)
 * @param C single precision monolish CRS Matrix (size M x N)
 * @note
 * - # of computation: MN
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
void sub(const matrix::CRS<float> &A, const matrix::CRS<float> &B,
         matrix::CRS<float> &C);

/**
 * @brief single precision element by element multiplication of CRS matrix A and
 * CRS matrix B.
 * @param A single precision monolish CRS Matrix (size M x N)
 * @param B single precision monolish CRS Matrix (size M x N)
 * @param C single precision monolish CRS Matrix (size M x N)
 * @note
 * - # of computation: MN
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
void mul(const matrix::CRS<float> &A, const matrix::CRS<float> &B,
         matrix::CRS<float> &C);

/**
 * @brief single precision element by element division of CRS matrix A and CRS
 * matrix B.
 * @param A single precision monolish CRS Matrix (size M x N)
 * @param B single precision monolish CRS Matrix (size M x N)
 * @param C single precision monolish CRS Matrix (size M x N)
 * @note
 * - # of computation: MN
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
void div(const matrix::CRS<float> &A, const matrix::CRS<float> &B,
         matrix::CRS<float> &C);

/**
 * @brief single precision scalar and Dence matrix addition (C[i][j] = A[i][j] +
 * alpha)
 * @param A single precision monolish CRS Matrix (size M x N)
 * @param alpha single precision scalar value
 * @param C single precision monolish CRS Matrix (size M x N)
 * @note
 * - # of computation: MN
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
void add(const matrix::CRS<float> &A, const float alpha, matrix::CRS<float> &C);

/**
 * @brief single precision scalar and Dence matrix subtraction (C[i][j] =
 * A[i][j] + alpha)
 * @param A single precision monolish CRS Matrix (size M x N)
 * @param alpha single precision scalar value
 * @param C single precision monolish CRS Matrix (size M x N)
 * @note
 * - # of computation: MN
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
void sub(const matrix::CRS<float> &A, const float alpha, matrix::CRS<float> &C);

/**
 * @brief single precision scalar and Dence matrix multiplication (C[i][j] =
 * A[i][j] + alpha)
 * @param A single precision monolish CRS Matrix (size M x N)
 * @param alpha single precision scalar value
 * @param C single precision monolish CRS Matrix (size M x N)
 * @note
 * - # of computation: MN
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
void mul(const matrix::CRS<float> &A, const float alpha, matrix::CRS<float> &C);

/**
 * @brief single precision scalar and Dence matrix division (C[i][j] = A[i][j] +
 * alpha)
 * @param A single precision monolish CRS Matrix (size M x N)
 * @param alpha single precision scalar value
 * @param C single precision monolish CRS Matrix (size M x N)
 * @note
 * - # of computation: MN
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
void div(const matrix::CRS<float> &A, const float alpha, matrix::CRS<float> &C);

} // namespace vml
} // namespace monolish
