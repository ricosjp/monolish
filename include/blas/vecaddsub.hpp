#pragma once
#include "../common/monolish_common.hpp"
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

/**
 * @name vecadd
 * @param a monolish vector or view1D (size N)
 * @param b monolish vector or view1D (size N)
 * @param y monolish vector or view1D (size N)
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
//@{
/**
 * @brief element by element addition of vector a and vector b.
 */
void vecadd(const vector<double> &a, const vector<double> &b,
            vector<double> &y);
void vecadd(const view1D<vector<double>, double> &a, const vector<double> &b,
            vector<double> &y);
void vecadd(const vector<double> &a, const view1D<vector<double>, double> &b,
            vector<double> &y);
void vecadd(const view1D<vector<double>, double> &a,
            const view1D<vector<double>, double> &b, vector<double> &y);
void vecadd(const vector<double> &a, const vector<double> &b,
            view1D<vector<double>, double> &y);
void vecadd(const view1D<vector<double>, double> &a, const vector<double> &b,
            view1D<vector<double>, double> &y);
void vecadd(const vector<double> &a, const view1D<vector<double>, double> &b,
            view1D<vector<double>, double> &y);
void vecadd(const view1D<vector<double>, double> &a,
            const view1D<vector<double>, double> &b,
            view1D<vector<double>, double> &y);
void vecadd(const vector<float> &a, const vector<float> &b, vector<float> &y);
void vecadd(const view1D<vector<float>, float> &a, const vector<float> &b,
            vector<float> &y);
void vecadd(const vector<float> &a, const view1D<vector<float>, float> &b,
            vector<float> &y);
void vecadd(const view1D<vector<float>, float> &a,
            const view1D<vector<float>, float> &b, vector<float> &y);
void vecadd(const vector<float> &a, const vector<float> &b,
            view1D<vector<float>, float> &y);
void vecadd(const view1D<vector<float>, float> &a, const vector<float> &b,
            view1D<vector<float>, float> &y);
void vecadd(const vector<float> &a, const view1D<vector<float>, float> &b,
            view1D<vector<float>, float> &y);
void vecadd(const view1D<vector<float>, float> &a,
            const view1D<vector<float>, float> &b,
            view1D<vector<float>, float> &y);
//@}

/**
 * @name vecsub
 * @param a monolish vector or view1D (size N)
 * @param b monolish vector or view1D (size N)
 * @param y monolish vector or view1D (size N)
 * @note
 * - # of computation: N
 * - Multi-threading: true
 * - GPU acceleration: true
 *    - # of data transfer: 0
 */
//@{
/**
 * @brief element by element subtraction of vector a and vector b.
 */
void vecsub(const vector<double> &a, const vector<double> &b,
            vector<double> &y);
void vecsub(const view1D<vector<double>, double> &a, const vector<double> &b,
            vector<double> &y);
void vecsub(const vector<double> &a, const view1D<vector<double>, double> &b,
            vector<double> &y);
void vecsub(const view1D<vector<double>, double> &a,
            const view1D<vector<double>, double> &b, vector<double> &y);
void vecsub(const vector<double> &a, const vector<double> &b,
            view1D<vector<double>, double> &y);
void vecsub(const view1D<vector<double>, double> &a, const vector<double> &b,
            view1D<vector<double>, double> &y);
void vecsub(const vector<double> &a, const view1D<vector<double>, double> &b,
            view1D<vector<double>, double> &y);
void vecsub(const view1D<vector<double>, double> &a,
            const view1D<vector<double>, double> &b,
            view1D<vector<double>, double> &y);
void vecsub(const vector<float> &a, const vector<float> &b, vector<float> &y);
void vecsub(const view1D<vector<float>, float> &a, const vector<float> &b,
            vector<float> &y);
void vecsub(const vector<float> &a, const view1D<vector<float>, float> &b,
            vector<float> &y);
void vecsub(const view1D<vector<float>, float> &a,
            const view1D<vector<float>, float> &b, vector<float> &y);
void vecsub(const vector<float> &a, const vector<float> &b,
            view1D<vector<float>, float> &y);
void vecsub(const view1D<vector<float>, float> &a, const vector<float> &b,
            view1D<vector<float>, float> &y);
void vecsub(const vector<float> &a, const view1D<vector<float>, float> &b,
            view1D<vector<float>, float> &y);
void vecsub(const view1D<vector<float>, float> &a,
            const view1D<vector<float>, float> &b,
            view1D<vector<float>, float> &y);
//@}

} // namespace blas
} // namespace monolish
