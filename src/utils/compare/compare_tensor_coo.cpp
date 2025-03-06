#include "../../../include/monolish/common/monolish_logger.hpp"
#include "../../../include/monolish/common/monolish_matrix.hpp"
#include "../../../include/monolish/common/monolish_tensor_dense.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {
namespace tensor {

template <typename T>
bool tensor_COO<T>::equal(const tensor_COO<T> &tens,
                          bool compare_cpu_and_device) const {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  if (shape.size() != tens.shape.size())
    logger.util_out();
  { return false; }
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
  }

  if (!(internal::vequal(get_nnz(), data(), tens.data(), false))) {
    logger.util_out();
    return false;
  }
  for (size_t i = 0; i < get_nnz(); ++i) {
    if (index[i].size() != tens.index[i].size()) {
      logger.util_out();
      return false;
    }
    if (!(internal::vequal(index[i].size(), index[i].data(),
                           tens.index[i].data(), false))) {
      logger.util_out();
      return false;
    }
  }

  logger.util_out();
  return true;
}

template bool tensor_COO<double>::equal(const tensor_COO<double> &tens,
                                        bool compare_cpu_and_device) const;
template bool tensor_COO<float>::equal(const tensor_COO<float> &tens,
                                       bool compare_cpu_and_device) const;

template <typename T>
bool tensor_COO<T>::operator==(const tensor_COO<T> &tens) const {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  bool ans = equal(tens, false);

  logger.util_out();
  return ans;
}
template bool
tensor_COO<double>::operator==(const tensor_COO<double> &tens) const;
template bool
tensor_COO<float>::operator==(const tensor_COO<float> &tens) const;

template <typename T>
bool tensor_COO<T>::operator!=(const tensor_COO<T> &tens) const {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  bool ans = equal(tens, false);

  logger.util_out();
  return !(ans);
}
template bool
tensor_COO<double>::operator!=(const tensor_COO<double> &tens) const;
template bool
tensor_COO<float>::operator!=(const tensor_COO<float> &tens) const;

} // namespace tensor
} // namespace monolish

namespace monolish {
namespace util {

template <typename T>
bool is_same_structure(const tensor::tensor_COO<T> &A,
                       const tensor::tensor_COO<T> &B) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  bool ans = true;

  if (A.get_shape() != B.get_shape()) {
    logger.util_out();
    ans = false;
  }
  for (size_t i = 0; i < A.get_nnz(); ++i) {
    if (!(internal::vequal(A.index[i].size(), A.index[i].data(),
                           B.index[i].data(), false))) {
      logger.util_out();
      return false;
    }
  }

  logger.util_out();
  return ans;
}

template bool is_same_structure(const tensor::tensor_COO<double> &A,
                                const tensor::tensor_COO<double> &B);
template bool is_same_structure(const tensor::tensor_COO<float> &A,
                                const tensor::tensor_COO<float> &B);

template <typename T>
bool is_same_size(const tensor::tensor_COO<T> &A,
                  const tensor::tensor_COO<T> &B) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  bool ans = true;

  if (A.get_shape() != B.get_shape()) {
    logger.util_out();
    ans = false;
  }

  logger.util_out();
  return ans;
}
template bool is_same_size(const tensor::tensor_COO<double> &A,
                           const tensor::tensor_COO<double> &B);
template bool is_same_size(const tensor::tensor_COO<float> &A,
                           const tensor::tensor_COO<float> &B);

} // namespace util
} // namespace monolish
