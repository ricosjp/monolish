#include "../../../include/monolish_blas.hpp"
#include "../../../include/monolish_equation.hpp"
#include "../../internal/monolish_internal.hpp"

#ifdef MONOLISH_USE_NVIDIA_GPU
#include "cusolverSp.h"
#endif

namespace monolish {

template <>
int equation::LU<matrix::CRS<double>, double>::cusolver_LU(
    matrix::CRS<double> &A, vector<double> &x, vector<double> &b) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);
  // nothing to do
  (void)(&A);
  (void)(&x);
  (void)(&b);
  if (1) {
    throw std::runtime_error("error sparse LU on GPU does not impl.");
  }

  logger.func_out();
  return 0;
}

} // namespace monolish
