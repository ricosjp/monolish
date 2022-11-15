#include "../../../include/monolish/common/monolish_dense.hpp"
#include "../../../include/monolish/common/monolish_logger.hpp"
#include "../../../include/monolish/common/monolish_matrix.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {
namespace matrix {

template <typename T> void COO<T>::transpose() {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  std::swap(rowN, colN);
  std::swap(row_index, col_index);

  logger.util_out();
}
template void COO<double>::transpose();
template void COO<float>::transpose();

template <typename T> void COO<T>::transpose(const COO &B) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  set_row(B.get_col());
  set_col(B.get_row());
  set_nnz(B.get_nnz());
  row_index = B.get_col_ind();
  col_index = B.get_row_ptr();
  val = B.get_val_ptr();
  logger.util_out();
}
template void COO<double>::transpose(const COO &B);
template void COO<float>::transpose(const COO &B);

} // namespace matrix
} // namespace monolish
