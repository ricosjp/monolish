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
  row_index = B.get_col_ind();
  col_index = B.get_row_ptr();
  auto val = B.get_val_ptr();
  if (get_device_mem_stat()) {
    if (get_nnz() != val.size()) {
      throw std::runtime_error(
          "Error: different nnz size GPU matrix cant use transpose");
    }
  } else {
    vad_create_flag = true;
    resize(val.size());
  }
  for (size_t i = 0; i < val.size(); ++i) {
    data()[i] = val[i];
  }
  logger.util_out();
}
template void COO<double>::transpose(const COO &B);
template void COO<float>::transpose(const COO &B);

} // namespace matrix
} // namespace monolish
