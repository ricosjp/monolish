#include "../../../../include/monolish_blas.hpp"
#include "../../../internal/monolish_internal.hpp"

namespace monolish {

namespace {
template <typename T>
void times_core(const T alpha, const matrix::CRS<T> &A, matrix::CRS<T> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_structure(A, C));
  assert(util::is_same_device_mem_stat(A, C));

  internal::vmul(A.get_nnz(), A.val.data(), alpha, C.val.data(),
                 A.get_device_mem_stat());

  logger.func_out();
}
} // namespace

namespace blas {
void times(const double alpha, const matrix::CRS<double> &A,
           matrix::CRS<double> &C) {
  times_core(alpha, A, C);
}
void times(const float alpha, const matrix::CRS<float> &A,
           matrix::CRS<float> &C) {
  times_core(alpha, A, C);
}

} // namespace blas
} // namespace monolish
