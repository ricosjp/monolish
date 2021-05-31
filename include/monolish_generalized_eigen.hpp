#pragma once
#include <omp.h>
#include <vector>

#include "./monolish_solver.hpp"
#include "common/monolish_common.hpp"

/**
 * @brief handling eigenvalues and eigenvectors
 **/
namespace monolish::generalized_eigen {

/**
 * @brief LOBPCG solver
 * @note
 * attribute:
 * - solver : true
 * - preconditioner : false
 * @note
 * input / archtecture
 * - Dense / Intel : true
 * - Dense / NVIDIA : true
 * - Dense / OSS : true
 * - Sparse / Intel : true
 * - Sparse / NVIDIA : true
 * - Sparse / OSS : true
 */
template <typename MATRIX, typename Float>
class LOBPCG : public solver::solver<MATRIX, Float> {
private:
  // TODO: support multiple lambda(eigenvalue)s
  int monolish_LOBPCG(MATRIX &A, MATRIX &B, vector<Float> &lambda,
                      matrix::Dense<Float> &x, int itype);

public:
  int solve(MATRIX &A, MATRIX &B, vector<Float> &lambda,
            matrix::Dense<Float> &x, int itype);

  void create_precond(MATRIX &A) {
    throw std::runtime_error("this precond. is not impl.");
  }

  void apply_precond(const vector<Float> &r, vector<Float> &z) {
    throw std::runtime_error("this precond. is not impl.");
  }

  /**
   * @brief get solver name "monolish::generalized_eigen::LOBPCG"
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  std::string name() const { return "monolish::generalized_eigen::LOBPCG"; }
};

/**
 * @brief Devide and Conquer solver
 * @note
 * attribute:
 * - solver : true
 * - preconditioner : false
 * @note
 * input / archtecture
 * - Dense / Intel : true
 * - Dense / NVIDIA : true
 * - Dense / OSS : true
 * - Sparse / Intel : false
 * - Sparse / NVIDIA : false
 * - Sparse / OSS : false
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

  /**
   * @brief get solver name "monolish::generalized_eigen::DC"
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  std::string name() const { return "monolish::generalized_eigen::DC"; }
};

} // namespace monolish::generalized_eigen
