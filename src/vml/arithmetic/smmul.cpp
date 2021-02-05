#include "../../../include/monolish_vml.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {

namespace {
template <typename F1, typename F2, typename F3> 
  void mmmul_core(const F1 &A, const F2 alpha, F3 &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(A, C));
  assert(util::is_same_device_mem_stat(A, C));

  internal::vmul(A.get_nnz(), A.val.data(), alpha, C.val.data(),
                 C.get_device_mem_stat());

  logger.func_out();
}

} // namespace

namespace vml {
// Dense //
void mul(const matrix::Dense<double> &A, const double alpha, matrix::Dense<double> &C) { 
  mmmul_core(A, alpha, C);
}

void mul(const matrix::Dense<float> &A, const float alpha, matrix::Dense<float> &C) { 
  mmmul_core(A, alpha, C);
}

// CRS //
void mul(const matrix::CRS<double> &A, const double alpha, matrix::CRS<double> &C) { 
  mmmul_core(A, alpha, C);
}

void mul(const matrix::CRS<float> &A, const float alpha, matrix::CRS<float> &C) { 
  mmmul_core(A, alpha, C);
}
} // namespace blas

} // namespace monolish
