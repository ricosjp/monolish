#include "../../../../include/monolish_blas.hpp"
#include "../../../internal/monolish_internal.hpp"

namespace monolish {

namespace {

// vector-tensor_row adds //
template <typename T, typename TENS1, typename VEC, typename TENS2>
void adds_row_core(const TENS1 &A, const size_t num, const VEC &x, TENS2 &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  assert(util::is_same_structure(A, C));
  assert(util::is_same_device_mem_stat(A, x, C));

  matrix::Dense<T> Amat, Cmat;
  Amat.move(A, -1, A.get_shape()[A.get_shape().size() - 1]);
  assert(Amat.get_col() == x.size());
  assert(Amat.get_row() >= num);
  Cmat.move(C, Amat.get_row(), Amat.get_col());

  blas::adds_row(Amat, num, x, Cmat);

  logger.func_out();
}

// vector-tensor_col adds //
template <typename T, typename TENS1, typename VEC, typename TENS2>
void adds_col_core(const TENS1 &A, const size_t num, const VEC &x, TENS2 &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  assert(util::is_same_structure(A, C));
  assert(util::is_same_device_mem_stat(A, x, C));

  matrix::Dense<T> Amat, Cmat;
  Amat.move(A, A.get_shape()[0], -1);
  assert(Amat.get_row() == x.size());
  assert(Amat.get_col() >= num);
  Cmat.move(C, Amat.get_row(), Amat.get_col());

  blas::adds_col(Amat, num, x, Cmat);

  logger.func_out();
}
} // namespace
} // namespace monolish
