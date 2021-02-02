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

#if MONOLISH_USE_GPU // gpu
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

  if (lib == 1) {
    std::vector<int> ipiv(std::min(A.get_row(), A.get_col()));
    internal::lapack::sytrf(A, ipiv);
    internal::lapack::sytrs(A, XB, ipiv);
  } else {
    logger.func_out();
    throw std::runtime_error("error solver.lib is not 1");
  }

  logger.func_out();
}
template <>
int equation::Cholesky<matrix::Dense<float>, float>::solve(
    matrix::Dense<float> &A, vector<float> &XB) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  if (lib == 1) {
    std::vector<int> ipiv(std::min(A.get_row(), A.get_col()));
    internal::lapack::sytrf(A, ipiv);
    internal::lapack::sytrs(A, XB, ipiv);
  } else {
    logger.func_out();
    throw std::runtime_error("error solver.lib is not 1");
  }

  logger.func_out();
}
} // namespace monolish
