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
 * Linear Algebra Package for Dense Matrix
 */
namespace lapack {

//////////////////////////////////////////////////////
//  Eigenvalue calculation
//////////////////////////////////////////////////////

/**
 * @brief Finds the eigenvalues and eigenvectors of a double precision dense
 * symmetric matrix A
 * @param jobz when 'N' then calculate eigenvalues only, and when 'V' calculate
 * eigenvalues and eigenvectors
 * @param uplo when 'U' then upper triangle of A is used, and when 'L' then
 * lower triangle of A is used
 * @param A double precision symmetric Dense matrix (size M x M)
 * @note
 * - # of computation: M^3
 * - A is destroyed after called
 * - Multi-threading: true
 * - GPU acceleration: false
 *    - # of data transfer: 0
 */
bool syev(const char* jobz, const char* uplo, matrix::Dense<double> &A,
          vector<double> &W);


} // namespace lapack
} // namespace monolish
