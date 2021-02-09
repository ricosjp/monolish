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
