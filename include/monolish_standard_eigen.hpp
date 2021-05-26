#pragma once
#include <omp.h>
#include <vector>

#include "./monolish_solver.hpp"
#include "common/monolish_common.hpp"

/**
 * @brief handling eigenvalues and eigenvectors
 **/
namespace monolish::standard_eigen {

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
  int monolish_LOBPCG(MATRIX &A, vector<Float> &lambda,
                      matrix::Dense<Float> &x);

public:
  /**
   * @brief calculate eigenvalues and eigenvectors or A by LOBPCG method(lib=0:
   *monolish)
   * @param[in] A CRS format Matrix
   * @param[in] lambda up to m smallest eigenvalue
   * @param[in] x corresponding eigenvectors in Dense matrix format
   * @return error code (only 0 now)
   **/
  int solve(MATRIX &A, vector<Float> &lambda, matrix::Dense<Float> &x);

  void create_precond(MATRIX &A) {
    throw std::runtime_error("this precond. is not impl.");
  }

  void apply_precond(const vector<Float> &r, vector<Float> &z) {
    throw std::runtime_error("this precond. is not impl.");
  }

  /**
   * @brief get solver name "monolish::standard_eigen::LOBPCG"
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  std::string name() const { return "monolish::standard_eigen::LOBPCG"; }
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
  int LAPACK_DC(MATRIX &A, vector<Float> &lambda);

public:
  int solve(MATRIX &A, vector<Float> &lambda);

  void create_precond(MATRIX &A) {
    throw std::runtime_error("this precond. is not impl.");
  }

  void apply_precond(const vector<Float> &r, vector<Float> &z) {
    throw std::runtime_error("this precond. is not impl.");
  }

  /**
   * @brief get solver name "monolish::standard_eigen::DC"
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  std::string name() const { return "monolish::standard_eigen::DC"; }
};
} // namespace monolish::standard_eigen
