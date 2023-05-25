#include "../../../../include/monolish_blas.hpp"
#include "../../../internal/monolish_internal.hpp"

namespace monolish {

namespace {

// scalar-tensor times //
template <typename T, typename TENS1, typename TENS2>
void times_core(const T alpha, const TENS1 &A, TENS2 &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  assert(util::is_same_structure(A, C));
  assert(util::is_same_device_mem_stat(A, C));

  internal::vmul(A.get_nnz(), A.data(), alpha, C.data(),
                 A.get_device_mem_stat());

  logger.func_out();
}

// vector-tensor_row times all //
template <typename T, typename TENS1, typename VEC, typename TENS2>
void times_row_core(const TENS1 &A, const VEC &x, TENS2 &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  assert(util::is_same_structure(A, C));
  assert(util::is_same_device_mem_stat(A, x, C));

  matrix::Dense<T> Amat, Cmat;
  Amat.move(A, -1, A.get_shape()[A.get_shape().size() - 1]);
  assert(Amat.get_col() == x.size());
  Cmat.move(C, Amat.get_row(), Amat.get_col());

  blas::times_row(Amat, x, Cmat);

  logger.func_out();
}

// vector-tensor_col times all //
template <typename T, typename TENS1, typename VEC, typename TENS2>
void times_col_core(const TENS1 &A, const VEC &x, TENS2 &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  assert(util::is_same_structure(A, C));
  assert(util::is_same_device_mem_stat(A, x, C));

  matrix::Dense<T> Amat, Cmat;
  Amat.move(A, A.get_shape()[0], -1);
  assert(Amat.get_row() == x.size());
  Cmat.move(C, Amat.get_row(), Amat.get_col());

  blas::times_col(Amat, x, Cmat);

  logger.func_out();
}
} // namespace
} // namespace monolish
