#include "../../../include/monolish_vml.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {

namespace {
template <typename F1, typename F2, typename F3>
void mmdiv_core(const F1 &A, const F2 &B, F3 &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(A, B, C));
  assert(util::is_same_structure(A, B, C));
  assert(util::is_same_device_mem_stat(A, B, C));

  internal::vdiv(A.get_nnz(), A.val.data(), B.val.data(), C.val.data(),
                 C.get_device_mem_stat());

  logger.func_out();
}

} // namespace

namespace vml {
// Dense //
void div(const matrix::Dense<double> &A, const matrix::Dense<double> &B,
         matrix::Dense<double> &C) {
  mmdiv_core(A, B, C);
}

void div(const matrix::Dense<float> &A, const matrix::Dense<float> &B,
         matrix::Dense<float> &C) {
  mmdiv_core(A, B, C);
}

// CRS //
void div(const matrix::CRS<double> &A, const matrix::CRS<double> &B,
         matrix::CRS<double> &C) {
  mmdiv_core(A, B, C);
}

void div(const matrix::CRS<float> &A, const matrix::CRS<float> &B,
         matrix::CRS<float> &C) {
  mmdiv_core(A, B, C);
}
} // namespace vml

} // namespace monolish
