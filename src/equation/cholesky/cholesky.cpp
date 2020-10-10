#include "../../../include/monolish_equation.hpp"
#include "../../monolish_internal.hpp"

namespace monolish {

template <typename T>
int equation::Cholesky<T>::solve(matrix::CRS<T> &A, vector<T> &x,
                                 vector<T> &b) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  int ret = -1;

#if USE_GPU // gpu
  if (lib == 1) {
    ret = cusolver_Cholesky(A, x, b);
  } else {
    logger.func_out();
    throw std::runtime_error("error solver.lib is not 1");
  }
  logger.func_out();
#else
  (void)(&A); (void)(&x); (void)(&b);
  throw std::runtime_error("error Cholesky on CPU does not impl.");
#endif
  logger.func_out();
  return ret;
}

template int equation::Cholesky<double>::solve(matrix::CRS<double> &A,
                                               vector<double> &x,
                                               vector<double> &b);
template int equation::Cholesky<float>::solve(matrix::CRS<float> &A,
                                              vector<float> &x,
                                              vector<float> &b);
} // namespace monolish
