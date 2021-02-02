#include "../../../include/monolish_equation.hpp"
#include "../../internal/lapack/monolish_lapack.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {

template <>
int equation::LU<matrix::CRS<double>, double>::solve(matrix::CRS<double> &A,
                                                     vector<double> &x,
                                                     vector<double> &b) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  (void)(&A);
  (void)(&x);
  (void)(&b);
  logger.func_out();
  throw std::runtime_error("error sparse Lu is not impl.");
}

template <>
int equation::LU<matrix::CRS<float>, float>::solve(matrix::CRS<float> &A,
                                                   vector<float> &x,
                                                   vector<float> &b) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  (void)(&A);
  (void)(&x);
  (void)(&b);
  logger.func_out();
  throw std::runtime_error("error sparse Lu is not impl.");
}

template <>
int equation::LU<matrix::Dense<double>, double>::solve(matrix::Dense<double> &A,
                                                       vector<double> &XB) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  if (lib == 1) {
    std::vector<int> ipiv(std::min(A.get_row(), A.get_col()));
    internal::lapack::getrf(A, ipiv);
    internal::lapack::getrs(A, XB, ipiv);
  } else {
    logger.func_out();
    throw std::runtime_error("error solver.lib is not 1");
  }

  logger.func_out();
}
template <>
int equation::LU<matrix::Dense<float>, float>::solve(matrix::Dense<float> &A,
                                                     vector<float> &XB) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  if (lib == 1) {
    std::vector<int> ipiv(std::min(A.get_row(), A.get_col()));
    internal::lapack::getrf(A, ipiv);
    internal::lapack::getrs(A, XB, ipiv);
  } else {
    logger.func_out();
    throw std::runtime_error("error solver.lib is not 1");
  }

  logger.func_out();
}

} // namespace monolish
