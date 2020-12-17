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

void tanh(const vector<double> &a, vector<double> &y);

//////////////////////////////////////////////////////
//  Vector
//////////////////////////////////////////////////////

/**
 * @brief double precision element by element addition of vector a and vector b.
 * @param a double precision monolish vector (size N)
 * @param b double precision monolish vector (size N)
 * @param y double precision monolish vector (size N)
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
void add(const vector<double> &a, const vector<double> &b, vector<double> &y);

/**
 * @brief double precision element by element subtraction of vector a and vector
 * b.
 * @param a double precision monolish vector (size N)
 * @param b double precision monolish vector (size N)
 * @param y double precision monolish vector (size N)
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
void sub(const vector<double> &a, const vector<double> &b, vector<double> &y);

/**
 * @brief double precision element by element multiplication of vector a and
 * vector b.
 * @param a double precision monolish vector (size N)
 * @param b double precision monolish vector (size N)
 * @param y double precision monolish vector (size N)
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
void mul(const vector<double> &a, const vector<double> &b, vector<double> &y);

/**
 * @brief double precision element by element division of vector a and vector b.
 * @param a double precision monolish vector (size N)
 * @param b double precision monolish vector (size N)
 * @param y double precision monolish vector (size N)
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
void div(const vector<double> &a, const vector<double> &b, vector<double> &y);

/**
 * @brief double precision scalar and vector addition (y[i] = a[i] + alpha)
 * @param a double precision monolish vector (size N)
 * @param alpha double precision scalar value
 * @param y double precision monolish vector (size N)
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
void add(const vector<double> &a, const double alpha, vector<double> &y);

/**
 * @brief double precision scalar and vector subtraction (y[i] = a[i] - alpha)
 * @param a double precision monolish vector (size N)
 * @param alpha double precision scalar value
 * @param y double precision monolish vector (size N)
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
void sub(const vector<double> &a, const double alpha, vector<double> &y);

/**
 * @brief double precision scalar and vector multiplication (y[i] = a[i] *
 * alpha)
 * @param a double precision monolish vector (size N)
 * @param alpha double precision scalar value
 * @param y double precision monolish vector (size N)
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
void mul(const vector<double> &a, const double alpha, vector<double> &y);

/**
 * @brief double precision scalar and vector division (y[i] = a[i] / alpha)
 * @param a double precision monolish vector (size N)
 * @param alpha double precision scalar value
 * @param y double precision monolish vector (size N)
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
void div(const vector<double> &a, const double alpha, vector<double> &y);

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
 */
void div(const matrix::CRS<double> &A, const double alpha,
         matrix::CRS<double> &C);

} // namespace vml
} // namespace monolish
