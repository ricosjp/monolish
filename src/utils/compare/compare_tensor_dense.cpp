#include "../../../include/monolish/common/monolish_logger.hpp"
#include "../../../include/monolish/common/monolish_tensor.hpp"
#include "../../../include/monolish/common/monolish_tensor_dense.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {
namespace tensor {

template <typename T>
bool tensor_Dense<T>::equal(const tensor_Dense<T> &tens,
                            bool compare_cpu_and_device) const {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  if (shape.size() != tens.shape.size()) {
    return false;
  }
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] != tens.shape[i]) {
      return false;
    }
  }
  if (get_device_mem_stat() != tens.get_device_mem_stat()) {
    return false;
  }

  if (get_device_mem_stat() == true) {
    if (!(internal::vequal(get_nnz(), data(), tens.data(), true))) {
      return false;
    }
  } else if (get_device_mem_stat() == false ||
             compare_cpu_and_device == false) {
    if (!(internal::vequal(get_nnz(), data(), tens.data(), false))) {
      return false;
    }
  }

  logger.util_out();
  return true;
}
template bool tensor_Dense<double>::equal(const tensor_Dense<double> &tens,
                                          bool compare_cpu_and_device) const;
template bool tensor_Dense<float>::equal(const tensor_Dense<float> &tens,
                                         bool compare_cpu_and_device) const;

template <typename T>
bool tensor_Dense<T>::operator==(const tensor_Dense<T> &tens) const {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  bool ans = equal(tens, false);

  logger.util_out();
  return ans;
}
template bool
tensor_Dense<double>::operator==(const tensor_Dense<double> &tens) const;
template bool
tensor_Dense<float>::operator==(const tensor_Dense<float> &tens) const;

template <typename T>
bool tensor_Dense<T>::operator!=(const tensor_Dense<T> &tens) const {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  bool ans = equal(tens, false);

  logger.util_out();
  return !(ans);
}
template bool
tensor_Dense<double>::operator!=(const tensor_Dense<double> &tens) const;
template bool
tensor_Dense<float>::operator!=(const tensor_Dense<float> &tens) const;

} // namespace tensor
} // namespace monolish

namespace monolish {
namespace util {

template <typename T>
bool is_same_structure(const tensor::tensor_Dense<T> &A,
                       const tensor::tensor_Dense<T> &B) {
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

template bool is_same_structure(const tensor::tensor_Dense<double> &A,
                                const tensor::tensor_Dense<double> &B);
template bool is_same_structure(const tensor::tensor_Dense<float> &A,
                                const tensor::tensor_Dense<float> &B);

template <typename T, typename U>
bool is_same_structure(const tensor::tensor_Dense<U> &A,
                       const view_tensor_Dense<T, U> &B) {
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

template bool
is_same_structure(const tensor::tensor_Dense<double> &A,
                  const view_tensor_Dense<vector<double>, double> &B);
template bool
is_same_structure(const tensor::tensor_Dense<double> &A,
                  const view_tensor_Dense<matrix::Dense<double>, double> &B);
template bool is_same_structure(
    const tensor::tensor_Dense<double> &A,
    const view_tensor_Dense<tensor::tensor_Dense<double>, double> &B);
template bool
is_same_structure(const tensor::tensor_Dense<float> &A,
                  const view_tensor_Dense<vector<float>, float> &B);
template bool
is_same_structure(const tensor::tensor_Dense<float> &A,
                  const view_tensor_Dense<matrix::Dense<float>, float> &B);
template bool is_same_structure(
    const tensor::tensor_Dense<float> &A,
    const view_tensor_Dense<tensor::tensor_Dense<float>, float> &B);

template <typename T, typename U>
bool is_same_structure(const view_tensor_Dense<T, U> &A,
                       const tensor::tensor_Dense<U> &B) {
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

template bool
is_same_structure(const view_tensor_Dense<vector<double>, double> &A,
                  const tensor::tensor_Dense<double> &B);
template bool
is_same_structure(const view_tensor_Dense<matrix::Dense<double>, double> &A,
                  const tensor::tensor_Dense<double> &B);
template bool is_same_structure(
    const view_tensor_Dense<tensor::tensor_Dense<double>, double> &A,
    const tensor::tensor_Dense<double> &B);
template bool
is_same_structure(const view_tensor_Dense<vector<float>, float> &A,
                  const tensor::tensor_Dense<float> &B);
template bool
is_same_structure(const view_tensor_Dense<matrix::Dense<float>, float> &A,
                  const tensor::tensor_Dense<float> &B);
template bool is_same_structure(
    const view_tensor_Dense<tensor::tensor_Dense<float>, float> &A,
    const tensor::tensor_Dense<float> &B);

template <typename T, typename U>
bool is_same_structure(const view_tensor_Dense<T, U> &A,
                       const view_tensor_Dense<T, U> &B) {
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

template bool
is_same_structure(const view_tensor_Dense<vector<double>, double> &A,
                  const view_tensor_Dense<vector<double>, double> &B);
template bool
is_same_structure(const view_tensor_Dense<vector<double>, double> &A,
                  const view_tensor_Dense<matrix::Dense<double>, double> &B);
template bool is_same_structure(
    const view_tensor_Dense<vector<double>, double> &A,
    const view_tensor_Dense<tensor::tensor_Dense<double>, double> &B);
template bool
is_same_structure(const view_tensor_Dense<matrix::Dense<double>, double> &A,
                  const view_tensor_Dense<vector<double>, double> &B);
template bool
is_same_structure(const view_tensor_Dense<matrix::Dense<double>, double> &A,
                  const view_tensor_Dense<matrix::Dense<double>, double> &B);
template bool is_same_structure(
    const view_tensor_Dense<matrix::Dense<double>, double> &A,
    const view_tensor_Dense<tensor::tensor_Dense<double>, double> &B);
template bool is_same_structure(
    const view_tensor_Dense<tensor::tensor_Dense<double>, double> &A,
    const view_tensor_Dense<vector<double>, double> &B);
template bool is_same_structure(
    const view_tensor_Dense<tensor::tensor_Dense<double>, double> &A,
    const view_tensor_Dense<matrix::Dense<double>, double> &B);
template bool is_same_structure(
    const view_tensor_Dense<tensor::tensor_Dense<double>, double> &A,
    const view_tensor_Dense<tensor::tensor_Dense<double>, double> &B);
template bool
is_same_structure(const view_tensor_Dense<vector<float>, float> &A,
                  const view_tensor_Dense<vector<float>, float> &B);
template bool
is_same_structure(const view_tensor_Dense<vector<float>, float> &A,
                  const view_tensor_Dense<matrix::Dense<float>, float> &B);
template bool is_same_structure(
    const view_tensor_Dense<vector<float>, float> &A,
    const view_tensor_Dense<tensor::tensor_Dense<float>, float> &B);
template bool
is_same_structure(const view_tensor_Dense<matrix::Dense<float>, float> &A,
                  const view_tensor_Dense<vector<float>, float> &B);
template bool
is_same_structure(const view_tensor_Dense<matrix::Dense<float>, float> &A,
                  const view_tensor_Dense<matrix::Dense<float>, float> &B);
template bool is_same_structure(
    const view_tensor_Dense<matrix::Dense<float>, float> &A,
    const view_tensor_Dense<tensor::tensor_Dense<float>, float> &B);
template bool is_same_structure(
    const view_tensor_Dense<tensor::tensor_Dense<float>, float> &A,
    const view_tensor_Dense<vector<float>, float> &B);
template bool is_same_structure(
    const view_tensor_Dense<tensor::tensor_Dense<float>, float> &A,
    const view_tensor_Dense<matrix::Dense<float>, float> &B);
template bool is_same_structure(
    const view_tensor_Dense<tensor::tensor_Dense<float>, float> &A,
    const view_tensor_Dense<tensor::tensor_Dense<float>, float> &B);

template <typename T>
bool is_same_size(const tensor::tensor_Dense<T> &A,
                  const tensor::tensor_Dense<T> &B) {
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

template bool is_same_size(const tensor::tensor_Dense<double> &A,
                           const tensor::tensor_Dense<double> &B);
template bool is_same_size(const tensor::tensor_Dense<float> &A,
                           const tensor::tensor_Dense<float> &B);

template <typename T, typename U>
bool is_same_size(const tensor::tensor_Dense<U> &A,
                  const view_tensor_Dense<T, U> &B) {
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

template bool is_same_size(const tensor::tensor_Dense<double> &A,
                           const view_tensor_Dense<vector<double>, double> &B);
template bool
is_same_size(const tensor::tensor_Dense<double> &A,
             const view_tensor_Dense<matrix::Dense<double>, double> &B);
template bool
is_same_size(const tensor::tensor_Dense<double> &A,
             const view_tensor_Dense<tensor::tensor_Dense<double>, double> &B);
template bool is_same_size(const tensor::tensor_Dense<float> &A,
                           const view_tensor_Dense<vector<float>, float> &B);
template bool
is_same_size(const tensor::tensor_Dense<float> &A,
             const view_tensor_Dense<matrix::Dense<float>, float> &B);
template bool
is_same_size(const tensor::tensor_Dense<float> &A,
             const view_tensor_Dense<tensor::tensor_Dense<float>, float> &B);

template <typename T, typename U>
bool is_same_size(const view_tensor_Dense<T, U> &A,
                  const tensor::tensor_Dense<U> &B) {
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

template bool is_same_size(const view_tensor_Dense<vector<double>, double> &A,
                           const tensor::tensor_Dense<double> &B);
template bool
is_same_size(const view_tensor_Dense<matrix::Dense<double>, double> &A,
             const tensor::tensor_Dense<double> &B);
template bool
is_same_size(const view_tensor_Dense<tensor::tensor_Dense<double>, double> &A,
             const tensor::tensor_Dense<double> &B);
template bool is_same_size(const view_tensor_Dense<vector<float>, float> &A,
                           const tensor::tensor_Dense<float> &B);
template bool
is_same_size(const view_tensor_Dense<matrix::Dense<float>, float> &A,
             const tensor::tensor_Dense<float> &B);
template bool
is_same_size(const view_tensor_Dense<tensor::tensor_Dense<float>, float> &A,
             const tensor::tensor_Dense<float> &B);

template <typename T, typename U>
bool is_same_size(const view_tensor_Dense<T, U> &A,
                  const view_tensor_Dense<T, U> &B) {
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

template bool is_same_size(const view_tensor_Dense<vector<double>, double> &A,
                           const view_tensor_Dense<vector<double>, double> &B);
template bool
is_same_size(const view_tensor_Dense<vector<double>, double> &A,
             const view_tensor_Dense<matrix::Dense<double>, double> &B);
template bool
is_same_size(const view_tensor_Dense<vector<double>, double> &A,
             const view_tensor_Dense<tensor::tensor_Dense<double>, double> &B);
template bool
is_same_size(const view_tensor_Dense<matrix::Dense<double>, double> &A,
             const view_tensor_Dense<vector<double>, double> &B);
template bool
is_same_size(const view_tensor_Dense<matrix::Dense<double>, double> &A,
             const view_tensor_Dense<matrix::Dense<double>, double> &B);
template bool
is_same_size(const view_tensor_Dense<matrix::Dense<double>, double> &A,
             const view_tensor_Dense<tensor::tensor_Dense<double>, double> &B);
template bool
is_same_size(const view_tensor_Dense<tensor::tensor_Dense<double>, double> &A,
             const view_tensor_Dense<vector<double>, double> &B);
template bool
is_same_size(const view_tensor_Dense<tensor::tensor_Dense<double>, double> &A,
             const view_tensor_Dense<matrix::Dense<double>, double> &B);
template bool
is_same_size(const view_tensor_Dense<tensor::tensor_Dense<double>, double> &A,
             const view_tensor_Dense<tensor::tensor_Dense<double>, double> &B);
template bool is_same_size(const view_tensor_Dense<vector<float>, float> &A,
                           const view_tensor_Dense<vector<float>, float> &B);
template bool
is_same_size(const view_tensor_Dense<vector<float>, float> &A,
             const view_tensor_Dense<matrix::Dense<float>, float> &B);
template bool
is_same_size(const view_tensor_Dense<vector<float>, float> &A,
             const view_tensor_Dense<tensor::tensor_Dense<float>, float> &B);
template bool
is_same_size(const view_tensor_Dense<matrix::Dense<float>, float> &A,
             const view_tensor_Dense<vector<float>, float> &B);
template bool
is_same_size(const view_tensor_Dense<matrix::Dense<float>, float> &A,
             const view_tensor_Dense<matrix::Dense<float>, float> &B);
template bool
is_same_size(const view_tensor_Dense<matrix::Dense<float>, float> &A,
             const view_tensor_Dense<tensor::tensor_Dense<float>, float> &B);
template bool
is_same_size(const view_tensor_Dense<tensor::tensor_Dense<float>, float> &A,
             const view_tensor_Dense<vector<float>, float> &B);
template bool
is_same_size(const view_tensor_Dense<tensor::tensor_Dense<float>, float> &A,
             const view_tensor_Dense<matrix::Dense<float>, float> &B);
template bool
is_same_size(const view_tensor_Dense<tensor::tensor_Dense<float>, float> &A,
             const view_tensor_Dense<tensor::tensor_Dense<float>, float> &B);

} // namespace util
} // namespace monolish
