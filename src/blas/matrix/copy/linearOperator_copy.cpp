#include "../../../../include/monolish_blas.hpp"
#include "../../../internal/monolish_internal.hpp"

namespace monolish {

namespace {
template <typename F>
void copy_core(const matrix::LinearOperator<F> &A,
               matrix::LinearOperator<F> &C) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  // err
  assert(util::is_same_size(A, C));
  assert(util::is_same_device_mem_stat(A, C));

  if (A.get_matvec_init_flag()) {
    C.set_matvec(A.get_matvec());
  }
  if (A.get_rmatvec_init_flag()) {
    C.set_rmatvec(A.get_rmatvec());
  }

  logger.util_out();
}
} // namespace

namespace blas {

void copy(const matrix::LinearOperator<double> &A,
          matrix::LinearOperator<double> &C) {
  copy_core(A, C);
}
void copy(const matrix::LinearOperator<float> &A,
          matrix::LinearOperator<float> &C) {
  copy_core(A, C);
}

} // namespace blas
} // namespace monolish
