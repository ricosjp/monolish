#pragma once
#include <omp.h>
#include <vector>

#include "./monolish_solver.hpp"
#include "monolish/common/monolish_common.hpp"

namespace monolish {
/**
 * @brief handling eigenvalues and eigenvectors
 **/
namespace generalized_eigen {

/**
 * @addtogroup gEigen
 * @{
 */

/**
 * \defgroup gLOBPCG monolish::standard_eigen::LOBPCG
 * @brief LOBPCG solver
 * @{
 */
/**
 * @brief LOBPCG solver
 * @note
 * attribute:
 * - solver : true
 * - preconditioner : false
 * @note
 * input / architecture
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
  [[nodiscard]] int monolish_LOBPCG(MATRIX &A, MATRIX &B, vector<Float> &lambda,
                                    matrix::Dense<Float> &x, int itype);

public:
  [[nodiscard]] int solve(MATRIX &A, MATRIX &B, vector<Float> &lambda,
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
  [[nodiscard]] std::string name() const {
    return "monolish::generalized_eigen::LOBPCG";
  }

  /**
   * @brief get solver name "LOBPCG"
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  [[nodiscard]] std::string solver_name() const { return "LOBPCG"; }
};
/**@}*/

/**
 * \defgroup gDC monolish::standard_eigen::DC
 * @brief Devide and Conquer solver
 * @{
 */
/**
 * @brief Devide and Conquer solver
 * @note
 * attribute:
 * - solver : true
 * - preconditioner : false
 * @note
 * input / architecture
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
  [[nodiscard]] int LAPACK_DC(MATRIX &A, MATRIX &B, vector<Float> &lambda,
                              int itype);

public:
  [[nodiscard]] int solve(MATRIX &A, MATRIX &B, vector<Float> &lambda,
                          int itype);

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
  [[nodiscard]] std::string name() const {
    return "monolish::generalized_eigen::DC";
  }

  /**
   * @brief get solver name "DC"
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  [[nodiscard]] std::string solver_name() const { return "DC"; }
};
/**@}*/
} // namespace generalized_eigen
} // namespace monolish
