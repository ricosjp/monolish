#include "../../../include/monolish/common/monolish_dense.hpp"
#include "../../../include/monolish/common/monolish_logger.hpp"
#include "../../../include/monolish/common/monolish_matrix.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {
namespace matrix {

template <typename T>
bool Dense<T>::equal(const Dense<T> &mat, bool compare_cpu_and_device) const {
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

  if (get_first() != mat.get_first()) {
    return false;
  }

  if (get_device_mem_stat() == true) {
    if (!(internal::vequal(get_nnz(), data(), mat.data(), true))) {
      return false;
    }
  } else if (get_device_mem_stat() == false ||
             compare_cpu_and_device == false) {
    if (!(internal::vequal(get_nnz(), data(), mat.data(), false))) {
      return false;
    }
  }

  logger.util_out();
  return true;
}
template bool Dense<double>::equal(const Dense<double> &mat,
                                   bool compare_cpu_and_device) const;
template bool Dense<float>::equal(const Dense<float> &mat,
                                  bool compare_cpu_and_device) const;

template <typename T> bool Dense<T>::operator==(const Dense<T> &mat) const {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  bool ans = equal(mat, false);

  logger.util_out();
  return ans;
}
template bool Dense<double>::operator==(const Dense<double> &mat) const;
template bool Dense<float>::operator==(const Dense<float> &mat) const;

template <typename T> bool Dense<T>::operator!=(const Dense<T> &mat) const {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  bool ans = equal(mat, false);

  logger.util_out();
  return !(ans);
}
template bool Dense<double>::operator!=(const Dense<double> &mat) const;
template bool Dense<float>::operator!=(const Dense<float> &mat) const;

} // namespace matrix
} // namespace monolish

namespace monolish {
namespace util {

template <typename T>
bool is_same_structure(const matrix::Dense<T> &A, const matrix::Dense<T> &B) {
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

template bool is_same_structure(const matrix::Dense<double> &A,
                                const matrix::Dense<double> &B);
template bool is_same_structure(const matrix::Dense<float> &A,
                                const matrix::Dense<float> &B);

template <typename T, typename U>
bool is_same_structure(const view_Dense<T, U> &A, const matrix::Dense<U> &B) {
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

template bool is_same_structure(const view_Dense<vector<double>, double> &A,
                                const matrix::Dense<double> &B);
template bool
is_same_structure(const view_Dense<matrix::Dense<double>, double> &A,
                  const matrix::Dense<double> &B);
template bool
is_same_structure(const view_Dense<tensor::tensor_Dense<double>, double> &A,
                  const matrix::Dense<double> &B);
template bool is_same_structure(const view_Dense<vector<float>, float> &A,
                                const matrix::Dense<float> &B);
template bool
is_same_structure(const view_Dense<matrix::Dense<float>, float> &A,
                  const matrix::Dense<float> &B);
template bool
is_same_structure(const view_Dense<tensor::tensor_Dense<float>, float> &A,
                  const matrix::Dense<float> &B);

template <typename T, typename U>
bool is_same_structure(const matrix::Dense<U> &A, const view_Dense<T, U> &B) {
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

template bool is_same_structure(const matrix::Dense<double> &A,
                                const view_Dense<vector<double>, double> &B);
template bool
is_same_structure(const matrix::Dense<double> &A,
                  const view_Dense<matrix::Dense<double>, double> &B);
template bool
is_same_structure(const matrix::Dense<double> &A,
                  const view_Dense<tensor::tensor_Dense<double>, double> &B);
template bool is_same_structure(const matrix::Dense<float> &A,
                                const view_Dense<vector<float>, float> &B);
template bool
is_same_structure(const matrix::Dense<float> &A,
                  const view_Dense<matrix::Dense<float>, float> &B);
template bool
is_same_structure(const matrix::Dense<float> &A,
                  const view_Dense<tensor::tensor_Dense<float>, float> &B);

template <typename T, typename U, typename V>
bool is_same_structure(const view_Dense<T, V> &A, const view_Dense<U, V> &B) {
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

template bool is_same_structure(const view_Dense<vector<double>, double> &A,
                                const view_Dense<vector<double>, double> &B);
template bool
is_same_structure(const view_Dense<vector<double>, double> &A,
                  const view_Dense<matrix::Dense<double>, double> &B);
template bool
is_same_structure(const view_Dense<vector<double>, double> &A,
                  const view_Dense<tensor::tensor_Dense<double>, double> &B);
template bool
is_same_structure(const view_Dense<matrix::Dense<double>, double> &A,
                  const view_Dense<vector<double>, double> &B);
template bool
is_same_structure(const view_Dense<matrix::Dense<double>, double> &A,
                  const view_Dense<matrix::Dense<double>, double> &B);
template bool
is_same_structure(const view_Dense<matrix::Dense<double>, double> &A,
                  const view_Dense<tensor::tensor_Dense<double>, double> &B);
template bool
is_same_structure(const view_Dense<tensor::tensor_Dense<double>, double> &A,
                  const view_Dense<vector<double>, double> &B);
template bool
is_same_structure(const view_Dense<tensor::tensor_Dense<double>, double> &A,
                  const view_Dense<matrix::Dense<double>, double> &B);
template bool
is_same_structure(const view_Dense<tensor::tensor_Dense<double>, double> &A,
                  const view_Dense<tensor::tensor_Dense<double>, double> &B);
template bool is_same_structure(const view_Dense<vector<float>, float> &A,
                                const view_Dense<vector<float>, float> &B);
template bool
is_same_structure(const view_Dense<vector<float>, float> &A,
                  const view_Dense<matrix::Dense<float>, float> &B);
template bool
is_same_structure(const view_Dense<vector<float>, float> &A,
                  const view_Dense<tensor::tensor_Dense<float>, float> &B);
template bool
is_same_structure(const view_Dense<matrix::Dense<float>, float> &A,
                  const view_Dense<vector<float>, float> &B);
template bool
is_same_structure(const view_Dense<matrix::Dense<float>, float> &A,
                  const view_Dense<matrix::Dense<float>, float> &B);
template bool
is_same_structure(const view_Dense<matrix::Dense<float>, float> &A,
                  const view_Dense<tensor::tensor_Dense<float>, float> &B);
template bool
is_same_structure(const view_Dense<tensor::tensor_Dense<float>, float> &A,
                  const view_Dense<vector<float>, float> &B);
template bool
is_same_structure(const view_Dense<tensor::tensor_Dense<float>, float> &A,
                  const view_Dense<matrix::Dense<float>, float> &B);
template bool
is_same_structure(const view_Dense<tensor::tensor_Dense<float>, float> &A,
                  const view_Dense<tensor::tensor_Dense<float>, float> &B);

template <typename T>
bool is_same_size(const matrix::Dense<T> &A, const matrix::Dense<T> &B) {
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

template bool is_same_size(const matrix::Dense<double> &A,
                           const matrix::Dense<double> &B);
template bool is_same_size(const matrix::Dense<float> &A,
                           const matrix::Dense<float> &B);

template <typename T, typename U>
bool is_same_size(const view_Dense<T, U> &A, const matrix::Dense<U> &B) {
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

template bool is_same_size(const view_Dense<vector<double>, double> &A,
                           const matrix::Dense<double> &B);
template bool is_same_size(const view_Dense<matrix::Dense<double>, double> &A,
                           const matrix::Dense<double> &B);
template bool
is_same_size(const view_Dense<tensor::tensor_Dense<double>, double> &A,
             const matrix::Dense<double> &B);
template bool is_same_size(const view_Dense<vector<float>, float> &A,
                           const matrix::Dense<float> &B);
template bool is_same_size(const view_Dense<matrix::Dense<float>, float> &A,
                           const matrix::Dense<float> &B);
template bool
is_same_size(const view_Dense<tensor::tensor_Dense<float>, float> &A,
             const matrix::Dense<float> &B);

template <typename T, typename U>
bool is_same_size(const matrix::Dense<U> &A, const view_Dense<T, U> &B) {
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

template bool is_same_size(const matrix::Dense<double> &A,
                           const view_Dense<vector<double>, double> &B);
template bool is_same_size(const matrix::Dense<double> &A,
                           const view_Dense<matrix::Dense<double>, double> &B);
template bool
is_same_size(const matrix::Dense<double> &A,
             const view_Dense<tensor::tensor_Dense<double>, double> &B);
template bool is_same_size(const matrix::Dense<float> &A,
                           const view_Dense<vector<float>, float> &B);
template bool is_same_size(const matrix::Dense<float> &A,
                           const view_Dense<matrix::Dense<float>, float> &B);
template bool
is_same_size(const matrix::Dense<float> &A,
             const view_Dense<tensor::tensor_Dense<float>, float> &B);

template <typename T, typename U, typename V>
bool is_same_size(const view_Dense<T, V> &A, const view_Dense<U, V> &B) {
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

template bool is_same_size(const view_Dense<vector<double>, double> &A,
                           const view_Dense<vector<double>, double> &B);
template bool is_same_size(const view_Dense<vector<double>, double> &A,
                           const view_Dense<matrix::Dense<double>, double> &B);
template bool
is_same_size(const view_Dense<vector<double>, double> &A,
             const view_Dense<tensor::tensor_Dense<double>, double> &B);
template bool is_same_size(const view_Dense<matrix::Dense<double>, double> &A,
                           const view_Dense<vector<double>, double> &B);
template bool is_same_size(const view_Dense<matrix::Dense<double>, double> &A,
                           const view_Dense<matrix::Dense<double>, double> &B);
template bool
is_same_size(const view_Dense<matrix::Dense<double>, double> &A,
             const view_Dense<tensor::tensor_Dense<double>, double> &B);
template bool
is_same_size(const view_Dense<tensor::tensor_Dense<double>, double> &A,
             const view_Dense<vector<double>, double> &B);
template bool
is_same_size(const view_Dense<tensor::tensor_Dense<double>, double> &A,
             const view_Dense<matrix::Dense<double>, double> &B);
template bool
is_same_size(const view_Dense<tensor::tensor_Dense<double>, double> &A,
             const view_Dense<tensor::tensor_Dense<double>, double> &B);
template bool is_same_size(const view_Dense<vector<float>, float> &A,
                           const view_Dense<vector<float>, float> &B);
template bool is_same_size(const view_Dense<vector<float>, float> &A,
                           const view_Dense<matrix::Dense<float>, float> &B);
template bool
is_same_size(const view_Dense<vector<float>, float> &A,
             const view_Dense<tensor::tensor_Dense<float>, float> &B);
template bool is_same_size(const view_Dense<matrix::Dense<float>, float> &A,
                           const view_Dense<vector<float>, float> &B);
template bool is_same_size(const view_Dense<matrix::Dense<float>, float> &A,
                           const view_Dense<matrix::Dense<float>, float> &B);
template bool
is_same_size(const view_Dense<matrix::Dense<float>, float> &A,
             const view_Dense<tensor::tensor_Dense<float>, float> &B);
template bool
is_same_size(const view_Dense<tensor::tensor_Dense<float>, float> &A,
             const view_Dense<vector<float>, float> &B);
template bool
is_same_size(const view_Dense<tensor::tensor_Dense<float>, float> &A,
             const view_Dense<matrix::Dense<float>, float> &B);
template bool
is_same_size(const view_Dense<tensor::tensor_Dense<float>, float> &A,
             const view_Dense<tensor::tensor_Dense<float>, float> &B);

} // namespace util
} // namespace monolish
