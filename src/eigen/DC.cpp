#include "../../include/monolish_blas.hpp"
#include "../../include/monolish_eigen.hpp"
#include "../internal/lapack/monolish_lapack.hpp"
#include "../internal/monolish_internal.hpp"

namespace monolish {

template <typename T>
int standard_eigen::DC<T>::LAPACK_DC(matrix::Dense<T> &A, vector<T> &lambda) {
  int ret = MONOLISH_SOLVER_SUCCESS;
  Logger &logger = Logger::get_instance();
  logger.solver_in(monolish_func);

  const char jobz = 'V';
  const char uplo = 'U';

  int info = internal::lapack::syevd(A, lambda, &jobz, &uplo);
  if (info > 0) {
    ret = MONOLISH_SOLVER_BREAKDOWN;
  } else if (info < 0) {
    ret = MONOLISH_SOLVER_RESIDUAL_NAN;
  }

  logger.solver_out();
  return ret;
}

template int standard_eigen::DC<double>::LAPACK_DC(matrix::Dense<double> &A,
                                                   vector<double> &lambda);
template int standard_eigen::DC<float>::LAPACK_DC(matrix::Dense<float> &A,
                                                  vector<float> &lambda);

template <typename T>
int standard_eigen::DC<T>::solve(matrix::Dense<T> &A, vector<T> &lambda) {
  Logger &logger = Logger::get_instance();
  logger.solver_in(monolish_func);

  int ret = 0;
  if (this->get_lib() == 0) {
    ret = LAPACK_DC(A, lambda);
  }

  logger.solver_out();
  return ret; // err code
}

template int standard_eigen::DC<double>::solve(matrix::Dense<double> &A,
                                               vector<double> &lambda);
template int standard_eigen::DC<float>::solve(matrix::Dense<float> &A,
                                              vector<float> &x);

} // namespace monolish
