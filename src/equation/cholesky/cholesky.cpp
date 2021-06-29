#include "../../../include/monolish_blas.hpp"
#include "../../../include/monolish_equation.hpp"
#include "../../internal/lapack/monolish_lapack.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {

template <typename MATRIX, typename T>
int equation::Cholesky<MATRIX, T>::solve(MATRIX &A, vector<T> &x,
                                         vector<T> &b) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  int ret = -1;

#if MONOLISH_USE_NVIDIA_GPU // gpu
  if (lib == 1) {
    ret = cusolver_Cholesky(A, x, b);
  } else {
    logger.func_out();
    throw std::runtime_error("error solver.lib is not 1");
  }
  logger.func_out();
#else
  (void)(&A);
  (void)(&x);
  (void)(&b);
  throw std::runtime_error("error Cholesky on CPU does not impl.");
#endif
  logger.func_out();
  return ret;
}

template int equation::Cholesky<matrix::CRS<double>, double>::solve(
    matrix::CRS<double> &A, vector<double> &x, vector<double> &b);
template int equation::Cholesky<matrix::CRS<float>, float>::solve(
    matrix::CRS<float> &A, vector<float> &x, vector<float> &b);

template <>
int equation::Cholesky<matrix::Dense<double>, double>::solve(
    matrix::Dense<double> &A, vector<double> &XB) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  if (A.get_device_mem_stat() == true) {
    throw std::runtime_error("error Dense Cholesky on GPU does not impl.");
  }

  int ret = MONOLISH_SOLVER_SUCCESS;

  if (lib == 1) {
    std::vector<int> ipiv(std::min(A.get_row(), A.get_col()));

    if (internal::lapack::sytrf(A, ipiv) != 0) {
      ret = MONOLISH_SOLVER_BREAKDOWN;
    };

    A.recv();
    XB.recv();

    if (internal::lapack::sytrs(A, XB, ipiv) != 0) {
      ret = MONOLISH_SOLVER_BREAKDOWN;
    }

    XB.send();

  } else {
    logger.func_out();
    throw std::runtime_error("error solver.lib is not 1");
  }

  logger.func_out();
  return ret;
}
template <>
int equation::Cholesky<matrix::Dense<float>, float>::solve(
    matrix::Dense<float> &A, vector<float> &XB) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  int ret = MONOLISH_SOLVER_SUCCESS;

  if (lib == 1) {
    std::vector<int> ipiv(std::min(A.get_row(), A.get_col()));

    if (internal::lapack::sytrf(A, ipiv) != 0) {
      ret = MONOLISH_SOLVER_BREAKDOWN;
    };

    A.recv();
    XB.recv();

    if (internal::lapack::sytrs(A, XB, ipiv) != 0) {
      ret = MONOLISH_SOLVER_BREAKDOWN;
    }

    XB.send();

  } else {
    logger.func_out();
    throw std::runtime_error("error solver.lib is not 1");
  }

  logger.func_out();
  return ret;
}

template <>
int equation::Cholesky<matrix::Dense<double>, double>::solve(
    matrix::Dense<double> &A, vector<double> &x, vector<double> &b) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  int ret = MONOLISH_SOLVER_SUCCESS;

  if (lib == 1) {
    std::vector<int> ipiv(std::min(A.get_row(), A.get_col()));
    monolish::blas::copy(b, x);

    if (internal::lapack::sytrf(A, ipiv) != 0) {
      ret = MONOLISH_SOLVER_BREAKDOWN;
    };

    A.recv();
    x.recv();

    if (internal::lapack::sytrs(A, x, ipiv) != 0) {
      ret = MONOLISH_SOLVER_BREAKDOWN;
    }

    x.send();

  } else {
    logger.func_out();
    throw std::runtime_error("error solver.lib is not 1");
  }

  logger.func_out();
  return ret;
}

template <>
int equation::Cholesky<matrix::Dense<float>, float>::solve(
    matrix::Dense<float> &A, vector<float> &x, vector<float> &b) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  int ret = MONOLISH_SOLVER_SUCCESS;

  if (lib == 1) {
    std::vector<int> ipiv(std::min(A.get_row(), A.get_col()));
    monolish::blas::copy(b, x);

    if (internal::lapack::sytrf(A, ipiv) != 0) {
      ret = MONOLISH_SOLVER_BREAKDOWN;
    };

    A.recv();
    x.recv();

    if (internal::lapack::sytrs(A, x, ipiv) != 0) {
      ret = MONOLISH_SOLVER_BREAKDOWN;
    }

    x.send();

  } else {
    logger.func_out();
    throw std::runtime_error("error solver.lib is not 1");
  }

  logger.func_out();
  return ret;
}
} // namespace monolish
