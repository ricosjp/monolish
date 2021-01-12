#include "../../../include/common/monolish_dense.hpp"
#include "../../../include/common/monolish_logger.hpp"
#include "../../../include/common/monolish_matrix.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {
namespace matrix {

template <typename T> bool Dense<T>::operator==(const Dense<T> &mat) const {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  if (get_row() != mat.get_row()) {
    return false;
  }
  if (get_col() != mat.get_col()) {
    return false;
  }

  if (get_device_mem_stat() != mat.get_device_mem_stat()) {
    return false;
  }

  if (get_device_mem_stat() == true) {
    if (!(internal::vequal(get_nnz(), val.data(), mat.val.data(), true))) {
      return false;
    }
  }

  if (!(internal::vequal(get_nnz(), val.data(), mat.val.data(), false))) {
    return false;
  }

  logger.util_out();
  return true;
}
template bool Dense<double>::operator==(const Dense<double> &mat) const;
template bool Dense<float>::operator==(const Dense<float> &mat) const;

template <typename T> bool Dense<T>::operator!=(const Dense<T> &mat) const {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  if (get_row() != mat.get_row()) {
    return true;
  }
  if (get_col() != mat.get_col()) {
    return true;
  }

  if (get_device_mem_stat() != mat.get_device_mem_stat()) {
    return true;
  }

  if (get_device_mem_stat() == true) {
    if (internal::vequal(get_nnz(), val.data(), mat.val.data(), true)) {
      return false;
    }
  }

  if (internal::vequal(get_nnz(), val.data(), mat.val.data(), false)) {
    return false;
  }

  logger.util_out();
  return true;
}
template bool Dense<double>::operator!=(const Dense<double> &mat) const;
template bool Dense<float>::operator!=(const Dense<float> &mat) const;

} // namespace matrix
} // namespace monolish
