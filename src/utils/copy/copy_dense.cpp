#include "../../../include/common/monolish_dense.hpp"
#include "../../../include/common/monolish_logger.hpp"
#include "../../../include/common/monolish_matrix.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {
namespace matrix {

template <typename T> void Dense<T>::operator=(const Dense<T> &mat) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  // err
  if (get_row() != mat.get_row()) {
    throw std::runtime_error("error A.row != mat.row");
  }
  if (get_col() != mat.get_col()) {
    throw std::runtime_error("error A.col != mat.col");
  }
  if (get_device_mem_stat() != mat.get_device_mem_stat()) {
    throw std::runtime_error("error get_device_mem_stat() is not same");
  }

  // gpu copy
  if (mat.get_device_mem_stat()) {
    internal::vcopy(get_nnz(), mat.val.data(), val.data(), true);
  } else {
    internal::vcopy(get_nnz(), mat.val.data(), val.data(), false);
  }

  logger.util_out();
}

template void Dense<double>::operator=(const Dense<double> &mat);
template void Dense<float>::operator=(const Dense<float> &mat);

} // namespace matrix
} // namespace monolish
