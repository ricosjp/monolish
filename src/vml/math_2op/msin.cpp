#include "../../../include/monolish_vml.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {
namespace {
template <typename F1, typename F2> void msin_core(const F1 &A, F2 &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(A, C));
  assert(util::is_same_structure(A, C));
  assert(util::is_same_device_mem_stat(A, C));

  internal::vsin(A.get_nnz(), A.val.data(), C.val.data(),
                 A.get_device_mem_stat());

  logger.func_out();
}
} // namespace

namespace vml {
void sin(const matrix::Dense<double> &A, matrix::Dense<double> &C) {
  msin_core(A, C);
}
void sin(const matrix::Dense<float> &A, matrix::Dense<float> &C) {
  msin_core(A, C);
}

void sin(const matrix::CRS<double> &A, matrix::CRS<double> &C) {
  msin_core(A, C);
}
void sin(const matrix::CRS<float> &A, matrix::CRS<float> &C) {
  msin_core(A, C);
}
} // namespace vml
} // namespace monolish
