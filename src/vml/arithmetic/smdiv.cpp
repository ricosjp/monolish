#include "../../../include/monolish_vml.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {

namespace {
template <typename F1, typename F2, typename F3>
void mmdiv_core(const F1 &A, const F2 alpha, F3 &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(A, C));
  assert(util::is_same_device_mem_stat(A, C));

  internal::vdiv(A.get_nnz(), A.val.data(), alpha, C.val.data(),
                 C.get_device_mem_stat());

  logger.func_out();
}

} // namespace

namespace vml {
// Dense //
void div(const matrix::Dense<double> &A, const double alpha,
         matrix::Dense<double> &C) {
  mmdiv_core(A, alpha, C);
}

void div(const matrix::Dense<float> &A, const float alpha,
         matrix::Dense<float> &C) {
  mmdiv_core(A, alpha, C);
}

// CRS //
void div(const matrix::CRS<double> &A, const double alpha,
         matrix::CRS<double> &C) {
  mmdiv_core(A, alpha, C);
}

void div(const matrix::CRS<float> &A, const float alpha,
         matrix::CRS<float> &C) {
  mmdiv_core(A, alpha, C);
}
} // namespace vml

} // namespace monolish
