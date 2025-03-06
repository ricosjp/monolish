#include "../../../include/monolish/common/monolish_dense.hpp"
#include "../../../include/monolish/common/monolish_logger.hpp"
#include "../../../include/monolish/common/monolish_matrix.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {
namespace tensor {

template <typename T>
bool tensor_CRS<T>::equal(const tensor_CRS<T> &tens,
                          bool compare_cpu_and_device) const {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  if (shape.size() != tens.shape.size()) {
    logger.util_out();
    return false;
  }
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] != tens.shape[i]) {
      logger.util_out();
      return false;
    }
  }
  if (get_device_mem_stat() != tens.get_device_mem_stat()) {
    logger.util_out();
    return false;
  }

  if (get_device_mem_stat() == true) {
    if (!(internal::vequal(get_nnz(), data(), tens.data(), true))) {
      logger.util_out();
      return false;
    }
    for (size_t i = 0; i < col_inds.size(); ++i) {
      if (!(internal::vequal(col_inds[i].size(), col_inds[i].data(),
                             tens.col_inds[i].data(), true))) {
        logger.util_out();
        return false;
      }
    }
    for (size_t i = 0; i < row_ptrs.size(); ++i) {
      if (!(internal::vequal(row_ptrs[i].size(), row_ptrs[i].data(),
                             tens.row_ptrs[i].data(), true))) {
        logger.util_out();
        return false;
      }
    }
  } else if (get_device_mem_stat() == false ||
             compare_cpu_and_device == false) {
    if (!(internal::vequal(get_nnz(), data(), tens.data(), false))) {
      logger.util_out();
      return false;
    }
    for (size_t i = 0; i < col_inds.size(); ++i) {
      if (!(internal::vequal(col_inds[i].size(), col_inds[i].data(),
                             tens.col_inds[i].data(), false))) {
        logger.util_out();
        return false;
      }
    }
    for (size_t i = 0; i < row_ptrs.size(); ++i) {
      if (!(internal::vequal(row_ptrs[i].size(), row_ptrs[i].data(),
                             tens.row_ptrs[i].data(), false))) {
        logger.util_out();
        return false;
      }
    }
  }

  logger.util_out();
  return true;
}
template bool tensor_CRS<double>::equal(const tensor_CRS<double> &tens,
                                        bool compare_cpu_and_device) const;
template bool tensor_CRS<float>::equal(const tensor_CRS<float> &tens,
                                       bool compare_cpu_and_device) const;

template <typename T>
bool tensor_CRS<T>::operator==(const tensor_CRS<T> &tens) const {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  bool ans = equal(tens, false);

  logger.util_out();
  return ans;
}
template bool
tensor_CRS<double>::operator==(const tensor_CRS<double> &tens) const;
template bool
tensor_CRS<float>::operator==(const tensor_CRS<float> &tens) const;

template <typename T>
bool tensor_CRS<T>::operator!=(const tensor_CRS<T> &tens) const {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  bool ans = equal(tens, false);

  logger.util_out();
  return !(ans);
}
template bool
tensor_CRS<double>::operator!=(const tensor_CRS<double> &tens) const;
template bool
tensor_CRS<float>::operator!=(const tensor_CRS<float> &tens) const;

} // namespace tensor
} // namespace monolish

namespace monolish {
namespace util {

template <typename T>
bool is_same_structure(const tensor::tensor_CRS<T> &A,
                       const tensor::tensor_CRS<T> &B) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  bool ans = true;

  if (A.get_shape().size() != B.get_shape().size()) {
    logger.util_out();
    return false;
  }
  for (size_t i = 0; i < A.get_shape().size(); ++i) {
    if (A.get_shape()[i] != B.get_shape()[i]) {
      logger.util_out();
      return false;
    }
  }

  if (A.get_hash() != B.get_hash()) {
    logger.util_out();
    return false;
  }

  logger.util_out();
  return ans;
}

template bool is_same_structure(const tensor::tensor_CRS<double> &A,
                                const tensor::tensor_CRS<double> &B);
template bool is_same_structure(const tensor::tensor_CRS<float> &A,
                                const tensor::tensor_CRS<float> &B);

template <typename T>
bool is_same_size(const tensor::tensor_CRS<T> &A,
                  const tensor::tensor_CRS<T> &B) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  bool ans = true;

  if (A.get_shape().size() != B.get_shape().size()) {
    logger.util_out();
    return false;
  }
  for (size_t i = 0; i < A.get_shape().size(); ++i) {
    if (A.get_shape()[i] != B.get_shape()[i]) {
      logger.util_out();
      return false;
    }
  }

  logger.util_out();
  return ans;
}

template bool is_same_size(const tensor::tensor_CRS<double> &A,
                           const tensor::tensor_CRS<double> &B);
template bool is_same_size(const tensor::tensor_CRS<float> &A,
                           const tensor::tensor_CRS<float> &B);

} // namespace util
} // namespace monolish
