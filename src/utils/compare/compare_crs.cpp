#include "../../../include/monolish/common/monolish_dense.hpp"
#include "../../../include/monolish/common/monolish_logger.hpp"
#include "../../../include/monolish/common/monolish_matrix.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {
namespace matrix {

template <typename T>
bool CRS<T>::equal(const CRS<T> &mat, bool compare_cpu_and_device) const {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

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
    if (!(internal::vequal(get_nnz(), col_ind.data(), mat.col_ind.data(),
                           true))) {
      logger.util_out();
      return false;
    }
    if (!(internal::vequal(get_nnz(), row_ptr.data(), mat.row_ptr.data(),
                           true))) {
      logger.util_out();
      return false;
    }
  } else if (get_device_mem_stat() == false ||
             compare_cpu_and_device == false) {
    if (!(internal::vequal(get_nnz(), val.data(), mat.val.data(), false))) {
      logger.util_out();
      return false;
    }
    if (!(internal::vequal(get_nnz(), col_ind.data(), mat.col_ind.data(),
                           false))) {
      logger.util_out();
      return false;
    }
    if (!(internal::vequal(get_nnz(), row_ptr.data(), mat.row_ptr.data(),
                           false))) {
      logger.util_out();
      return false;
    }
  }

  logger.util_out();
  return true;
}
template bool CRS<double>::equal(const CRS<double> &mat,
                                 bool compare_cpu_and_device) const;
template bool CRS<float>::equal(const CRS<float> &mat,
                                bool compare_cpu_and_device) const;

template <typename T> bool CRS<T>::operator==(const CRS<T> &mat) const {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  bool ans = equal(mat, false);

  logger.util_out();
  return ans;
}
template bool CRS<double>::operator==(const CRS<double> &mat) const;
template bool CRS<float>::operator==(const CRS<float> &mat) const;

template <typename T> bool CRS<T>::operator!=(const CRS<T> &mat) const {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  bool ans = equal(mat, false);

  logger.util_out();
  return !(ans);
}
template bool CRS<double>::operator!=(const CRS<double> &mat) const;
template bool CRS<float>::operator!=(const CRS<float> &mat) const;

} // namespace matrix
} // namespace monolish

namespace monolish {
namespace util {

template <typename T>
bool is_same_structure(const matrix::CRS<T> &A, const matrix::CRS<T> &B) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  bool ans = true;

  if (A.get_row() != B.get_row() || A.get_col() != B.get_col()) {
    logger.util_out();
    ans = false;
  }

  if (A.get_hash() != B.get_hash()) {
    logger.util_out();
    return false;
  }

  logger.util_out();
  return ans;
}

template bool is_same_structure(const matrix::CRS<double> &A,
                                const matrix::CRS<double> &B);
template bool is_same_structure(const matrix::CRS<float> &A,
                                const matrix::CRS<float> &B);

template <typename T>
bool is_same_size(const matrix::CRS<T> &A, const matrix::CRS<T> &B) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  bool ans = true;

  if (A.get_row() != B.get_row() || A.get_col() != B.get_col()) {
    logger.util_out();
    ans = false;
  }

  logger.util_out();
  return ans;
}

template bool is_same_size(const matrix::CRS<double> &A,
                           const matrix::CRS<double> &B);
template bool is_same_size(const matrix::CRS<float> &A,
                           const matrix::CRS<float> &B);

} // namespace util
} // namespace monolish
