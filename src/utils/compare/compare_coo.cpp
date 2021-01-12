#include "../../../include/common/monolish_dense.hpp"
#include "../../../include/common/monolish_logger.hpp"
#include "../../../include/common/monolish_matrix.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {
namespace matrix {

template <typename T> bool COO<T>::equal(const COO<T> &mat, bool compare_cpu_and_device) const {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  if (get_device_mem_stat()) {
    throw std::runtime_error("Error, GPU COO cant use operator==");
  }

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
    if (!(internal::vequal(get_nnz(), col_index.data(), mat.col_index.data(),
                           true))) {
      return false;
    }
    if (!(internal::vequal(get_nnz(), row_index.data(), mat.row_index.data(),
                           true))) {
      return false;
    }
  }

  if (!(internal::vequal(get_nnz(), val.data(), mat.val.data(), false))) {
    return false;
  }
  if (!(internal::vequal(get_nnz(), col_index.data(), mat.col_index.data(),
                         false))) {
    return false;
  }
  if (!(internal::vequal(get_nnz(), row_index.data(), mat.row_index.data(),
                         false))) {
    return false;
  }

  logger.util_out();
  return true;
}
template bool COO<double>::equal(const COO<double> &mat, bool compare_cpu_and_device) const;
template bool COO<float>::equal(const COO<float> &mat, bool compare_cpu_and_device) const;

template <typename T> bool COO<T>::operator==(const COO<T> &mat) const {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  bool ret = equal(mat);

  logger.util_out();
  return ret;
}
template bool COO<double>::operator==(const COO<double> &mat) const;
template bool COO<float>::operator==(const COO<float> &mat) const;

template <typename T> bool COO<T>::operator!=(const COO<T> &mat) const {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  bool ret = equal(mat);

  logger.util_out();
  return !(true);
}
template bool COO<double>::operator!=(const COO<double> &mat) const;
template bool COO<float>::operator!=(const COO<float> &mat) const;

} // namespace matrix
} // namespace monolish
