#pragma once
#include <vector>

#if defined USE_MPI
#include <mpi.h>
#endif

#include "./monolish_solver.hpp"
#include "common/monolish_common.hpp"
#include <functional>

namespace monolish {
namespace equation {

/**
 * @brief none solver class
 */
template <typename MATRIX, typename Float>
class none : public monolish::solver::solver<MATRIX, Float> {
public:
  void create_precond(MATRIX &A);
  void apply_precond(const vector<Float> &r, vector<Float> &z);
  int solve(MATRIX &A, vector<Float> &x, vector<Float> &b);
};

/**
 * @brief CG solver class
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
};

/**
 * @brief BiCGSTAB solver class
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
};

/**
 * @brief Jacobi solver class
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
};

/**
 * @brief LU solver class (does not impl. now)
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
};

/**
 * @brief QR solver class (GPU only now). can use set_tol(), get_til(),
 * set_reorder(), get_singularity(). default reorder algorithm is csrmetisnd
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
};

/**
 * @brief Cholesky solver class (GPU only now). can use set_tol(), get_til(),
 * set_reorder(), get_singularity(). default reorder algorithm is csrmetisnd
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
};
} // namespace equation
} // namespace monolish
