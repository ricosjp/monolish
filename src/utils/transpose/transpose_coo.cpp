#include "../../../include/common/monolish_dense.hpp"
#include "../../../include/common/monolish_logger.hpp"
#include "../../../include/common/monolish_matrix.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {
namespace matrix {

template <typename T> COO<T> &COO<T>::transpose() {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  using std::swap;
  swap(rowN, colN);
  swap(row_index, col_index);
  return *this;
  logger.util_out();
}
template COO<double> &COO<double>::transpose();
template COO<float> &COO<float>::transpose();

template <typename T> void COO<T>::transpose(COO &B) const {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  B.set_row(get_col());
  B.set_col(get_row());
  B.set_nnz(get_nnz());
  B.row_index = get_col_ind();
  B.col_index = get_row_ptr();
  B.val = get_val_ptr();
  logger.util_out();
}
template void COO<double>::transpose(COO &B) const;
template void COO<float>::transpose(COO &B) const;

} // namespace matrix
} // namespace monolish
