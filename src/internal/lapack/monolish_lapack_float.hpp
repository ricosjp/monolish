#pragma once
#include "../../../include/common/monolish_common.hpp"
#include <stdio.h>
#include <vector>

#if defined USE_MPI
#include <mpi.h>
#endif

namespace monolish {
namespace internal {
/**
 * @brief
 * Linear Algebra Package for Dense Matrix
 */
namespace lapack {

//////////////////////////////////////////////////////
//  Eigenvalue calculation
//////////////////////////////////////////////////////

/**
 * @brief Finds the eigenvalues and eigenvectors of a single precision dense
 * symmetric matrix A
 * @param jobz when 'N' then calculate eigenvalues only, and when 'V' calculate
 * eigenvalues and eigenvectors
 * @param uplo when 'U' then upper triangle of A is used, and when 'L' then
 * lower triangle of A is used
 * @param A single precision symmetric Dense matrix (size M x M)
 * @note
 * - # of computation: M^3
 * - A is destroyed after called
 * - Multi-threading: true
 * - GPU acceleration: false
 *    - # of data transfer: 0
 */
bool syev(const char *jobz, const char *uplo, matrix::Dense<float> &A,
          vector<float> &W);

/**
 * @brief Finds the eigenvalues and eigenvectors of a single precision dense
 * symmetric matrix A
 * @param jobz when 'N' then calculate eigenvalues only, and when 'V' calculate
 * eigenvalues and eigenvectors
 * @param uplo when 'U' then upper triangle of A is used, and when 'L' then
 * lower triangle of A is used
 * @param A single precision symmetric Dense matrix (size M x M)
 * @param B single precision symmetric Dense Matrix (size M x M)
 * @note
 * - # of computation: M^3
 * - A is destroyed after called
 * - Multi-threading: true
 * - GPU acceleration: false
 *    - # of data transfer: 0
 */
bool sygv(const int itype, const char *jobz, const char *uplo,
          matrix::Dense<float> &A, matrix::Dense<float> &B,
          vector<float> &W);

} // namespace lapack
} // namespace internal
} // namespace monolish
