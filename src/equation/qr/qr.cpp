#include "../../../include/monolish_equation.hpp"
#include "../../monolish_internal.hpp"

namespace monolish {
template <typename T>
int equation::QR<T>::solve(matrix::CRS<T> &A, vector<T> &x, vector<T> &b) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  int ret = -1;

#if USE_GPU // gpu
  if (lib == 1) {
    ret = cusolver_QR(A, x, b);
  } else {
    logger.func_out();
    throw std::runtime_error("error solver.lib is not 1");
  }
#else
  logger.func_out();
  throw std::runtime_error("error QR on CPU does not impl.");
#endif

  logger.func_out();
  return ret;
}
template int equation::QR<double>::solve(matrix::CRS<double> &A,
                                         vector<double> &x, vector<double> &b);
template int equation::QR<float>::solve(matrix::CRS<float> &A, vector<float> &x,
                                        vector<float> &b);
} // namespace monolish
