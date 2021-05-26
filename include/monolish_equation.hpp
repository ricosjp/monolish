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
  int solve(MATRIX &A, vector<Float> &x, vector<Float> &b);

  /**
   * @brief get solver name "monolish::equation::none"
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  std::string name() const { return "monolish::equation::none"; }
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
  int monolish_CG(MATRIX &A, vector<Float> &x, vector<Float> &b);

public:
  /**
   * @brief solve Ax = b by BiCGSTAB method(lib=0: monolish)
   * @param[in] A CRS format Matrix
   * @param[in] x solution vector
   * @param[in] b right hand vector
   * @return error code (only 0 now)
   **/
  int solve(MATRIX &A, vector<Float> &x, vector<Float> &b);

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
  std::string name() const { return "monolish::equation::CG"; }
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
  int monolish_BiCGSTAB(MATRIX &A, vector<Float> &x, vector<Float> &b);

public:
  /**
   * @brief solve Ax = b by BiCGSTAB method (lib=0: monolish)
   * @param[in] A CRS format Matrix
   * @param[in] x solution vector
   * @param[in] b right hand vector
   * @return error code (only 0 now)
   **/
  int solve(MATRIX &A, vector<Float> &x, vector<Float> &b);

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
  std::string name() const { return "monolish::equation::BiCGSTAB"; }
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
  int monolish_Jacobi(MATRIX &A, vector<Float> &x, vector<Float> &b);

public:
  /**
   * @brief solve Ax = b by jacobi method(lib=0: monolish)
   * @param[in] A CRS format Matrix
   * @param[in] x solution vector
   * @param[in] b right hand vector
   * @return error code (only 0 now)
   **/
  int solve(MATRIX &A, vector<Float> &x, vector<Float> &b);
  void create_precond(MATRIX &A);
  void apply_precond(const vector<Float> &r, vector<Float> &z);

  /**
   * @brief get solver name "monolish::equation::Jacobi"
   * @note
   * - # of computation: 1
   * - Multi-threading: false
   * - GPU acceleration: false
   **/
  std::string name() const { return "monolish::equation::Jacobi"; }
};

/**
 * @brief LU solver class (Dense, CPU only now)
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
  int mumps_LU(MATRIX &A, vector<double> &x, vector<double> &b);
  int cusolver_LU(MATRIX &A, vector<double> &x, vector<double> &b);
  int singularity;
  int reorder = 3;

public:
  void set_reorder(int r) { reorder = r; }
  int get_sigularity() { return singularity; }
  int solve(MATRIX &A, vector<Float> &x, vector<Float> &b);
  int solve(MATRIX &A, vector<Float> &xb);
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
  std::string name() const { return "monolish::equation::LU"; }
};

/**
 * @brief QR solver class (Dense, GPU only now). can use set_tol(), get_tol(),
 * set_reorder(), get_singularity().
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
  int cusolver_QR(MATRIX &A, vector<double> &x, vector<double> &b);
  int cusolver_QR(MATRIX &A, vector<float> &x, vector<float> &b);
  int singularity;
  int reorder = 3;

public:
  /**
   * @brief 0: no ordering 1: symrcm, 2: symamd, 3: csrmetisnd is used to reduce
   * zero fill-in.
   */
  void set_reorder(int r) { reorder = r; }

  /**
   * @brief -1 if A is symmetric postive definite.
   * default reorder algorithm is csrmetisnd
   */
  int get_sigularity() { return singularity; }

  /**
   * @brief solve Ax=b
   */
  int solve(MATRIX &A, vector<Float> &x, vector<Float> &b);
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
  std::string name() const { return "monolish::equation::QR"; }
};

/**
 * @brief Cholesky solver class.
 * It can use set_tol(), get_tol(), set_reorder(), get_singularity().
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
  int cusolver_Cholesky(MATRIX &A, vector<float> &x, vector<float> &b);
  int cusolver_Cholesky(MATRIX &A, vector<double> &x, vector<double> &b);
  int singularity;
  int reorder = 3;

public:
  /**
   * @brief 0: no ordering 1: symrcm, 2: symamd, 3: csrmetisnd is used to reduce
   * zero fill-in.
   * default reorder algorithm is csrmetisnd.
   */
  void set_reorder(int r) { reorder = r; }

  /**
   * @brief -1 if A is symmetric postive definite.
   */
  int get_sigularity() { return singularity; }

  /**
   * @brief solve Ax=b
   */
  int solve(MATRIX &A, vector<Float> &x, vector<Float> &b);
  int solve(MATRIX &A, vector<Float> &xb);

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
  std::string name() const { return "monolish::equation::Cholesky"; }
};
} // namespace monolish::equation
