#pragma once
#include <omp.h>
#include <vector>

#if defined USE_MPI
#include <mpi.h>
#endif

#include "./monolish_solver.hpp"
#include "common/monolish_common.hpp"

namespace monolish {

/**
 * @brief handling eigenvalues and eigenvectors
 **/
namespace standard_eigen {

/**
 * @brief LOBPCG solver
 */
template <typename MATRIX, typename Float> class LOBPCG : public solver::solver<MATRIX, Float> {
private:
  // TODO: support multiple lambda(eigenvalue)s
  int monolish_LOBPCG(MATRIX &A, Float &lambda, vector<Float> &x);

public:
  /**
   * @brief calculate eigenvalues and eigenvectors or A by LOBPCG method(lib=0:
   *monolish)
   * @param[in] A CRS format Matrix
   * @param[in] lambda smallest eigenvalue
   * @param[in] x corresponding eigenvector
   * @return error code (only 0 now)
   **/
  int solve(MATRIX &A, Float &lambda, vector<Float> &x);

  void create_precond(MATRIX &A) {
    throw std::runtime_error("this precond. is not impl.");
  }

  void apply_precond(const vector<Float> &r, vector<Float> &z) {
    throw std::runtime_error("this precond. is not impl.");
  }
};

/**
 * @brief Devide and Conquer solver
 */
template <typename MATRIX, typename Float> class DC : public solver::solver<MATRIX, Float> {
private:
  int LAPACK_DC(MATRIX &A, vector<Float> &lambda);

public:
  int solve(MATRIX &A, vector<Float> &lambda);

  void create_precond(MATRIX &A) {
    throw std::runtime_error("this precond. is not impl.");
  }

  void apply_precond(const vector<Float> &r, vector<Float> &z) {
    throw std::runtime_error("this precond. is not impl.");
  }
};

} // namespace standard_eigen
} // namespace monolish
