#include "../../../include/monolish_equation.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {
template <typename MATRIX, typename T>
int equation::QR<MATRIX, T>::solve(MATRIX &A, vector<T> &x, vector<T> &b) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  int ret = -1;

#if MONOLISH_USE_NVIDIA_GPU // gpu
  if (lib == 1) {
    ret = cusolver_QR(A, x, b);
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
template int equation::QR<matrix::CRS<double>, double>::solve(
    matrix::CRS<double> &A, vector<double> &x, vector<double> &b);
template int equation::QR<matrix::CRS<float>, float>::solve(
    matrix::CRS<float> &A, vector<float> &x, vector<float> &b);
} // namespace monolish
