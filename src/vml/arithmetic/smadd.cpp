#include "../../../include/monolish_vml.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {

namespace {
template <typename F1, typename F2, typename F3> 
  void mmadd_core(const F1 &A, const F2 alpha, F3 &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(A, C));
  assert(util::is_same_device_mem_stat(A, C));

  internal::vadd(A.get_nnz(), A.val.data(), alpha, C.val.data(),
                 C.get_device_mem_stat());

  logger.func_out();
}

} // namespace

namespace vml {
// Dense //
void add(const matrix::Dense<double> &A, const double alpha, matrix::Dense<double> &C) { 
  mmadd_core(A, alpha, C);
}

void add(const matrix::Dense<float> &A, const float alpha, matrix::Dense<float> &C) { 
  mmadd_core(A, alpha, C);
}

// CRS //
void add(const matrix::CRS<double> &A, const double alpha, matrix::CRS<double> &C) { 
  mmadd_core(A, alpha, C);
}

void add(const matrix::CRS<float> &A, const float alpha, matrix::CRS<float> &C) { 
  mmadd_core(A, alpha, C);
}
} // namespace blas

} // namespace monolish
