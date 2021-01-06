#pragma once
#include <omp.h>
#include <vector>

#if defined USE_MPI
#include <mpi.h>
#endif

#include "common/monolish_common.hpp"

#include "monolish_equation.hpp"

namespace monolish {

/**
 * @brief handling eigenvalues and eigenvectors
 **/
namespace eigen {

/**
 * @brief LOBPCG solver
 */
template <typename Float> class LOBPCG : public equation::solver<Float> {
private:
  // TODO: support multiple lambda(eigenvalue)s
  int monolish_LOBPCG(matrix::CRS<Float> const &A, Float &lambda,
                      vector<Float> &x);

public:
  /**
   * @brief calculate eigenvalues and eigenvectors or A by LOBPCG method(lib=0:
   *monolish)
   * @param[in] A CRS format Matrix
   * @param[in] lambda smallest eigenvalue
   * @param[in] x corresponding eigenvector
   * @return error code (only 0 now)
   **/
  int solve(matrix::CRS<Float> const &A, Float &lambda, vector<Float> &x);

  void create_precond(matrix::CRS<Float> &A) {
    throw std::runtime_error("this precond. is not impl.");
  }

  void apply_precond(const vector<Float> &r, vector<Float> &z) {
    throw std::runtime_error("this precond. is not impl.");
  }
};

} // namespace eigen
} // namespace monolish
