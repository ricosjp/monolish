#pragma once
#include <vector>

#include "./monolish_solver.hpp"
#include "monolish/common/monolish_common.hpp"
#include <functional>

namespace monolish {
/**
 * @brief
 * Linear equation solvers for Dense and sparse matrix
 */
namespace equation {

/**
 * @addtogroup equations
 * @{
 */

/**
 * \defgroup none monolish::equation::none
 * @brief none solver (nothing to do)
 * @{
 */
/**
 * @brief none solver class(nothing to do)
 * @note
 * attribute:
 * - solver : true
 * - preconditioner : true
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
class none : public monolish::solver::solver<MATRIX, Float> {
public:
  void create_precond(MATRIX &A);
  void apply_precond(const vector<Float> &r, vector<Float> &z);
  [[nodiscard]] int solve(MATRIX &A, vector<Float> &x, vector<Float> &b);

  /**
   * @brief get solver name "monolish::equation::none"
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  [[nodiscard]] std::string name() const { return "monolish::equation::none"; }

  /**
   * @brief get solver name "none"
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  [[nodiscard]] std::string solver_name() const { return "none"; }
};
/**@}*/

/**
 * \defgroup cg monolish::equation::CG
 * @brief CG solver
 * @{
 */
/**
 * @brief CG solver class
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
class CG : public monolish::solver::solver<MATRIX, Float> {
private:
  [[nodiscard]] int monolish_CG(MATRIX &A, vector<Float> &x, vector<Float> &b);

public:
  /**
   * @brief solve Ax = b by BiCGSTAB method(lib=0: monolish)
   * @param[in] A CRS format Matrix
   * @param[in] x solution vector
   * @param[in] b right hand vector
   * @return error code, see
   *https://github.com/ricosjp/monolish/blob/master/include/common/monolish_common.hpp
   **/
  [[nodiscard]] int solve(MATRIX &A, vector<Float> &x, vector<Float> &b);

  void create_precond(MATRIX &A) {
    throw std::runtime_error("this precond. is not impl.");
  }

  void apply_precond(const vector<Float> &r, vector<Float> &z) {
    throw std::runtime_error("this precond. is not impl.");
  }

  /**
   * @brief get solver name "monolish::equation::CG"
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  [[nodiscard]] std::string name() const { return "monolish::equation::CG"; }

  /**
   * @brief get solver name "CG"
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  [[nodiscard]] std::string solver_name() const { return "CG"; }
};
/**@}*/

/**
 * \defgroup bicgstab monolish::equation::BiCGSTAB
 * @brief BiCGSTAB solver
 * @{
 */

/**
 * @brief BiCGSTAB solver class
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
class BiCGSTAB : public monolish::solver::solver<MATRIX, Float> {
private:
  [[nodiscard]] int monolish_BiCGSTAB(MATRIX &A, vector<Float> &x,
                                      vector<Float> &b);

public:
  /**
   * @brief solve Ax = b by BiCGSTAB method (lib=0: monolish)
   * @param[in] A CRS format Matrix
   * @param[in] x solution vector
   * @param[in] b right hand vector
   * @return error code, see
   *https://github.com/ricosjp/monolish/blob/master/include/common/monolish_common.hpp
   **/
  [[nodiscard]] int solve(MATRIX &A, vector<Float> &x, vector<Float> &b);

  void create_precond(MATRIX &A) {
    throw std::runtime_error("this precond. is not impl.");
  }

  void apply_precond(const vector<Float> &r, vector<Float> &z) {
    throw std::runtime_error("this precond. is not impl.");
  }

  /**
   * @brief get solver name "monolish::equation::BiCGSTAB"
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  [[nodiscard]] std::string name() const {
    return "monolish::equation::BiCGSTAB";
  }

  /**
   * @brief get solver name "BiCGSTAB"
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  [[nodiscard]] std::string solver_name() const { return "BiCGSTAB"; }
};
/**@}*/

/**
 * \defgroup jacobi monolish::equation::Jacobi
 * @brief Jacobi solver class
 * @{
 */
/**
 * @brief Jacobi solver class
 * @note
 * attribute:
 * - solver : true
 * - preconditioner : true
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
class Jacobi : public monolish::solver::solver<MATRIX, Float> {
private:
  [[nodiscard]] int monolish_Jacobi(MATRIX &A, vector<Float> &x,
                                    vector<Float> &b);

public:
  /**
   * @brief solve Ax = b by jacobi method(lib=0: monolish)
   * @param[in] A CRS format Matrix
   * @param[in] x solution vector
   * @param[in] b right hand vector
   * @return error code, see
   *https://github.com/ricosjp/monolish/blob/master/include/common/monolish_common.hpp
   **/
  [[nodiscard]] int solve(MATRIX &A, vector<Float> &x, vector<Float> &b);
  void create_precond(MATRIX &A);
  void apply_precond(const vector<Float> &r, vector<Float> &z);

  /**
   * @brief get solver name "monolish::equation::Jacobi"
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  [[nodiscard]] std::string name() const {
    return "monolish::equation::Jacobi";
  }

  /**
   * @brief get solver name "Jacobi"
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  [[nodiscard]] std::string solver_name() const { return "Jacobi"; }
};
/**@}*/

/**
 * \defgroup sor monolish::equation::SOR
 * @brief SOR solver class
 * @{
 */
/**
 * @brief SOR solver class
 * @note
 * attribute:
 * - solver : true
 * - preconditioner : true
 * @note
 * input / architecture
 * - Dense / Intel : true
 * - Dense / NVIDIA : true
 * - Dense / OSS : true
 * - Sparse / Intel : true
 * - Sparse / NVIDIA : true
 * - Sparse / OSS : true
 * @warning
 * SOR is not completely parallelized.
 * The part of solving the lower triangular matrix is performed sequentially.
 * On the GPU, one vector is received before the lower triangular matrix solving
 * process, and one vector is sent after the process.
 */
template <typename MATRIX, typename Float>
class SOR : public monolish::solver::solver<MATRIX, Float> {
private:
  [[nodiscard]] int monolish_SOR(MATRIX &A, vector<Float> &x, vector<Float> &b);

public:
  /**
   * @brief solve Ax = b by SOR method(lib=0: monolish)
   * @param[in] A CRS format Matrix
   * @param[in] x solution vector
   * @param[in] b right hand vector
   * @return error code, see
   *https://github.com/ricosjp/monolish/blob/master/include/common/monolish_common.hpp
   * @warning
   * SOR is not completely parallelized.
   * The part of solving the lower triangular matrix is performed sequentially.
   * On the GPU, one vector is received before the lower triangular matrix
   *solving process, and one vector is sent after the process.
   **/
  [[nodiscard]] int solve(MATRIX &A, vector<Float> &x, vector<Float> &b);
  void create_precond(MATRIX &A);
  void apply_precond(const vector<Float> &r, vector<Float> &z);

  /**
   * @brief get solver name "monolish::equation::SOR"
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  [[nodiscard]] std::string name() const { return "monolish::equation::SOR"; }

  /**
   * @brief get solver name "SOR"
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  [[nodiscard]] std::string solver_name() const { return "SOR"; }
};
/**@}*/

/**
 * \defgroup IC monolish::equation::IC
 * @brief Incomplete Cholesky solver class
 * @{
 */
/**
 * @brief Incomplete Cholesky solver class
 * @note
 * attribute:
 * - solver : true
 * - preconditioner : false
 * @note
 * input / architecture
 * - Dense / Intel : false
 * - Dense / NVIDIA : false
 * - Dense / OSS : false
 * - Sparse / Intel : false
 * - Sparse / NVIDIA : true
 * - Sparse / OSS : false
 */
template <typename MATRIX, typename Float>
class IC : public monolish::solver::solver<MATRIX, Float> {
private:
  int cusparse_IC(MATRIX &A, vector<Float> &x, vector<Float> &b);
  void *matM = 0, *matL = 0;
  void *infoM = 0, *infoL = 0, *infoLt = 0;
  void *cusparse_handle = nullptr;
  int bufsize;
  monolish::vector<double> buf;
  monolish::vector<Float> zbuf;

public:
  ~IC();
  /**
   * @brief solve with incomplete Cholesky factorization
   * @warning
   * This solves Ax = b incompletely. In many cases the answer is wrong.
   **/
  [[nodiscard]] int solve(MATRIX &A, vector<Float> &x, vector<Float> &b);
  void create_precond(MATRIX &A);
  void apply_precond(const vector<Float> &r, vector<Float> &z);

  /**
   * @brief get solver name "monolish::equation::IC"
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  std::string name() const { return "monolish::equation::IC"; }

  /**
   * @brief get solver name "IC"
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  std::string solver_name() const { return "IC"; }
};
/**@}*/

/**
 * \defgroup ILU monolish::equation::ILU
 * @brief Incomplete LU solver class
 * @{
 */
/**
 * @brief Incomplete LU solver class
 * @note
 * attribute:
 * - solver : true
 * - preconditioner : false
 * @note
 * input / architecture
 * - Dense / Intel : false
 * - Dense / NVIDIA : false
 * - Dense / OSS : false
 * - Sparse / Intel : false
 * - Sparse / NVIDIA : true
 * - Sparse / OSS : false
 */
template <typename MATRIX, typename Float>
class ILU : public monolish::solver::solver<MATRIX, Float> {
private:
  int cusparse_ILU(MATRIX &A, vector<Float> &x, vector<Float> &b);
  void *matM = 0, *matL = 0, *matU = 0;
  void *infoM = 0, *infoL = 0, *infoU = 0;
  void *cusparse_handle = nullptr;
  int bufsize;
  monolish::vector<double> buf;
  monolish::vector<Float> zbuf;

public:
  ~ILU();
  /**
   * @brief solve with incomplete LU factorization
   * @warning
   * This solves Ax = b incompletely. In many cases the answer is wrong.
   **/
  [[nodiscard]] int solve(MATRIX &A, vector<Float> &x, vector<Float> &b);
  void create_precond(MATRIX &A);
  void apply_precond(const vector<Float> &r, vector<Float> &z);

  /**
   * @brief get solver name "monolish::equation::ILU"
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  std::string name() const { return "monolish::equation::ILU"; }

  /**
   * @brief get solver name "ILU"
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  std::string solver_name() const { return "ILU"; }
};
/**@}*/

/**
 * \defgroup LU monolish::equation::LU
 * @brief LU solver class
 * @{
 */
/**
 * @brief LU solver class
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
class LU : public monolish::solver::solver<MATRIX, Float> {
private:
  int lib = 1; // lib is 1
  [[nodiscard]] int mumps_LU(MATRIX &A, vector<double> &x, vector<double> &b);
  [[nodiscard]] int cusolver_LU(MATRIX &A, vector<double> &x,
                                vector<double> &b);

public:
  [[nodiscard]] int solve(MATRIX &A, vector<Float> &x, vector<Float> &b);
  [[nodiscard]] int solve(MATRIX &A, vector<Float> &xb);

  void create_precond(MATRIX &A) {
    throw std::runtime_error("this precond. is not impl.");
  }
  void apply_precond(const vector<Float> &r, vector<Float> &z) {
    throw std::runtime_error("this precond. is not impl.");
  }

  /**
   * @brief get solver name "monolish::equation::LU"
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  [[nodiscard]] std::string name() const { return "monolish::equation::LU"; }

  /**
   * @brief get solver name "LU"
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  [[nodiscard]] std::string solver_name() const { return "LU"; }
};
/**@}*/

/**
 * \defgroup QR monolish::equation::QR
 * @brief QR solver class
 * @{
 */
/**
 * @brief QR solver class.
 * @note
 * attribute:
 * - solver : true
 * - preconditioner : false
 * @note
 * input / architecture
 * - Dense / Intel : false
 * - Dense / NVIDIA : false
 * - Dense / OSS : false
 * - Sparse / Intel : false
 * - Sparse / NVIDIA : true
 * - Sparse / OSS : false
 */
template <typename MATRIX, typename Float>
class QR : public monolish::solver::solver<MATRIX, Float> {
private:
  int lib = 1; // lib is 1
  [[nodiscard]] int cusolver_QR(MATRIX &A, vector<double> &x,
                                vector<double> &b);
  [[nodiscard]] int cusolver_QR(MATRIX &A, vector<float> &x, vector<float> &b);

public:
  /**
   * @brief solve Ax=b
   */
  [[nodiscard]] int solve(MATRIX &A, vector<Float> &x, vector<Float> &b);
  void create_precond(MATRIX &A) {
    throw std::runtime_error("this precond. is not impl.");
  }
  void apply_precond(const vector<Float> &r, vector<Float> &z) {
    throw std::runtime_error("this precond. is not impl.");
  }

  /**
   * @brief get solver name "monolish::equation::QR"
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  [[nodiscard]] std::string name() const { return "monolish::equation::QR"; }

  /**
   * @brief get solver name "QR"
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  [[nodiscard]] std::string solver_name() const { return "QR"; }
};
/**@}*/

/**
 * \defgroup chol monolish::equation::Cholesky
 * @brief Cholesky solver class.
 * @{
 */
/**
 * @brief Cholesky solver class.
 * @note
 * attribute:
 * - solver : true
 * - preconditioner : false
 * @note
 * input / architecture
 * - Dense / Intel : true
 * - Dense / NVIDIA : false
 * - Dense / OSS : true
 * - Sparse / Intel : false
 * - Sparse / NVIDIA : true
 * - Sparse / OSS : false
 */
template <typename MATRIX, typename Float>
class Cholesky : public monolish::solver::solver<MATRIX, Float> {
private:
  int lib = 1; // lib is 1
  [[nodiscard]] int cusolver_Cholesky(MATRIX &A, vector<float> &x,
                                      vector<float> &b);
  [[nodiscard]] int cusolver_Cholesky(MATRIX &A, vector<double> &x,
                                      vector<double> &b);

public:
  /**
   * @brief solve Ax=b
   */
  [[nodiscard]] int solve(MATRIX &A, vector<Float> &x, vector<Float> &b);
  [[nodiscard]] int solve(MATRIX &A, vector<Float> &xb);

  void create_precond(matrix::CRS<Float> &A) {
    throw std::runtime_error("this precond. is not impl.");
  }
  void apply_precond(const vector<Float> &r, vector<Float> &z) {
    throw std::runtime_error("this precond. is not impl.");
  }

  /**
   * @brief get solver name "monolish::equation::Cholesky"
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  [[nodiscard]] std::string name() const {
    return "monolish::equation::Cholesky";
  }

  /**
   * @brief get solver name "Cholesky"
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  [[nodiscard]] std::string solver_name() const { return "Cholesky"; }
};
/**@}*/
/**@}*/

} // namespace equation
} // namespace monolish
