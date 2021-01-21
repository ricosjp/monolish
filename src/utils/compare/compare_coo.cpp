#include "../../../include/common/monolish_dense.hpp"
#include "../../../include/common/monolish_logger.hpp"
#include "../../../include/common/monolish_matrix.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {
namespace matrix {

template <typename T>
bool COO<T>::equal(const COO<T> &mat, bool compare_cpu_and_device) const {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  if (get_device_mem_stat()) {
    throw std::runtime_error("Error, GPU COO cant use operator==");
  }

  if (get_row() != mat.get_row()) {
    logger.util_out();
    return false;
  }
  if (get_col() != mat.get_col()) {
    logger.util_out();
    return false;
  }

  if (get_device_mem_stat() != mat.get_device_mem_stat()) {
    logger.util_out();
    return false;
  }

  if (get_device_mem_stat() == true) {
    if (!(internal::vequal(get_nnz(), val.data(), mat.val.data(), true))) {
      logger.util_out();
      return false;
    }
    if (!(internal::vequal(get_nnz(), col_index.data(), mat.col_index.data(),
                           true))) {
      logger.util_out();
      return false;
    }
    if (!(internal::vequal(get_nnz(), row_index.data(), mat.row_index.data(),
                           true))) {
      logger.util_out();
      return false;
    }
  }

  if (!(internal::vequal(get_nnz(), val.data(), mat.val.data(), false))) {
    logger.util_out();
    return false;
  }
  if (!(internal::vequal(get_nnz(), col_index.data(), mat.col_index.data(),
                         false))) {
    logger.util_out();
    return false;
  }
  if (!(internal::vequal(get_nnz(), row_index.data(), mat.row_index.data(),
                         false))) {
    logger.util_out();
    return false;
  }

  logger.util_out();
  return true;
}
template bool COO<double>::equal(const COO<double> &mat,
                                 bool compare_cpu_and_device) const;
template bool COO<float>::equal(const COO<float> &mat,
                                bool compare_cpu_and_device) const;

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

namespace monolish {
namespace util {
template <typename T>
bool is_same_structure(matrix::COO<T> A, matrix::COO<T> B) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  bool ans = true;

  if (A.get_row() != B.get_row() && A.get_col() != B.get_col()) {
    logger.util_out();
    ans = false;
  }

  if (!(internal::vequal(A.get_nnz(), A.col_index.data(), B.col_index.data(),
                         false))) {
    logger.util_out();
    return false;
  }
  if (!(internal::vequal(A.get_nnz(), A.row_index.data(), B.row_index.data(),
                         false))) {
    logger.util_out();
    return false;
  }

  logger.util_out();
  return ans;
}

template bool is_same_structure(matrix::COO<double> A, matrix::COO<double> B);
template bool is_same_structure(matrix::COO<float> A, matrix::COO<float> B);

} // namespace util
} // namespace monolish
