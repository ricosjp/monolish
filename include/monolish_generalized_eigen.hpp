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
namespace generalized_eigen {

/**
 * @brief LOBPCG solver
 */
template <typename MATRIX, typename Float>
class LOBPCG : public solver::solver<MATRIX, Float> {
private:
  // TODO: support multiple lambda(eigenvalue)s
  int monolish_LOBPCG(MATRIX &A, MATRIX &B, vector<Float> &lambda, matrix::Dense<Float> &x,
                      int itype);

public:
  int solve(MATRIX &A, MATRIX &B, vector<Float> &lambda, matrix::Dense<Float> &x, int itype);

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
template <typename MATRIX, typename Float>
class DC : public solver::solver<MATRIX, Float> {
private:
  int LAPACK_DC(MATRIX &A, MATRIX &B, vector<Float> &lambda, int itype);

public:
  int solve(MATRIX &A, MATRIX &B, vector<Float> &lambda, int itype);

  void create_precond(MATRIX &A) {
    throw std::runtime_error("this precond. is not impl.");
  }

  void apply_precond(const vector<Float> &r, vector<Float> &z) {
    throw std::runtime_error("this precond. is not impl.");
  }
};

} // namespace generalized_eigen
} // namespace monolish
