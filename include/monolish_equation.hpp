#pragma once
#include <vector>

#include "./monolish_solver.hpp"
#include "common/monolish_common.hpp"
#include <functional>

/**
 * @brief
 * Linear equation solvers for Dense and sparse matrix
 * Scalar
 */
namespace monolish::equation {

/**
 * @brief none solver class(nothing to do)
 * @note
 * attribute:
 * - solver : true
 * - preconditioner : true
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

/**
 * @brief CG solver class
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

/**
 * @brief BiCGSTAB solver class
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

/**
 * @brief Jacobi solver class
 * @note
 * attribute:
 * - solver : true
 * - preconditioner : true
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

/**
 * @brief SOR solver class
 * @note
 * attribute:
 * - solver : true
 * - preconditioner : true
 * @note
 * input / archtecture
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

/**
 * @brief Incomplete LU solver class
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
class ILU : public monolish::solver::solver<MATRIX, Float> {
private:
  int cusparse_ILU(MATRIX &A, vector<Float> &x, vector<Float> &b);
  void *matM=0, *matL=0, *matU=0;
  void *infoM=0, *infoL=0, *infoU=0;

public:
  int solve(MATRIX &A, vector<Float> &x, vector<Float> &b);
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

/**
 * @brief LU solver class
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

/**
 * @brief QR solver class.
 * @note
 * attribute:
 * - solver : true
 * - preconditioner : false
 * @note
 * input / archtecture
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

/**
 * @brief Cholesky solver class.
 * @note
 * attribute:
 * - solver : true
 * - preconditioner : false
 * @note
 * input / archtecture
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

} // namespace monolish::equation
