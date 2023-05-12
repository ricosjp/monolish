#include "../../../include/monolish/common/monolish_common.hpp"
#include "../../../include/monolish/common/monolish_dense.hpp"
#include "../../../include/monolish/common/monolish_logger.hpp"
#include "../../../include/monolish/common/monolish_matrix.hpp"
#include "../../../include/monolish/common/monolish_tensor.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {

template <typename T, typename U>
void view_tensor_Dense<T, U>::operator=(const tensor::tensor_Dense<U> &tens) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  // err
  assert(monolish::util::is_same_size(*this, tens));
  assert(monolish::util::is_same_device_mem_stat(*this, tens));

  // gpu copy
  if (tens.get_device_mem_stat()) {
    internal::vcopy(get_nnz(), tens.begin(), begin(), true);
  } else {
    internal::vcopy(get_nnz(), tens.begin(), begin(), false);
  }

  logger.util_out();
}

template void view_tensor_Dense<vector<double>, double>::operator=(
    const tensor::tensor_Dense<double> &tens);
template void view_tensor_Dense<matrix::Dense<double>, double>::operator=(
    const tensor::tensor_Dense<double> &tens);
template void
view_tensor_Dense<tensor::tensor_Dense<double>, double>::operator=(
    const tensor::tensor_Dense<double> &tens);
template void view_tensor_Dense<vector<float>, float>::operator=(
    const tensor::tensor_Dense<float> &tens);
template void view_tensor_Dense<matrix::Dense<float>, float>::operator=(
    const tensor::tensor_Dense<float> &tens);
template void view_tensor_Dense<tensor::tensor_Dense<float>, float>::operator=(
    const tensor::tensor_Dense<float> &tens);

template <typename T, typename U>
void view_tensor_Dense<T, U>::operator=(
    const view_tensor_Dense<vector<U>, U> &tens) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  // err
  assert(monolish::util::is_same_size(*this, tens));
  assert(monolish::util::is_same_device_mem_stat(*this, tens));

  // gpu copy
  if (tens.get_device_mem_stat()) {
    internal::vcopy(get_nnz(), tens.begin(), begin(), true);
  } else {
    internal::vcopy(get_nnz(), tens.begin(), begin(), false);
  }

  logger.util_out();
}

template void view_tensor_Dense<vector<double>, double>::operator=(
    const view_tensor_Dense<vector<double>, double> &tens);
template void view_tensor_Dense<matrix::Dense<double>, double>::operator=(
    const view_tensor_Dense<vector<double>, double> &tens);
template void
view_tensor_Dense<tensor::tensor_Dense<double>, double>::operator=(
    const view_tensor_Dense<vector<double>, double> &tens);
template void view_tensor_Dense<vector<float>, float>::operator=(
    const view_tensor_Dense<vector<float>, float> &tens);
template void view_tensor_Dense<matrix::Dense<float>, float>::operator=(
    const view_tensor_Dense<vector<float>, float> &tens);
template void view_tensor_Dense<tensor::tensor_Dense<float>, float>::operator=(
    const view_tensor_Dense<vector<float>, float> &tens);

template <typename T, typename U>
void view_tensor_Dense<T, U>::operator=(
    const view_tensor_Dense<matrix::Dense<U>, U> &tens) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  // err
  assert(monolish::util::is_same_size(*this, tens));
  assert(monolish::util::is_same_device_mem_stat(*this, tens));

  // gpu copy
  if (tens.get_device_mem_stat()) {
    internal::vcopy(get_nnz(), tens.begin(), begin(), true);
  } else {
    internal::vcopy(get_nnz(), tens.begin(), begin(), false);
  }

  logger.util_out();
}

template void view_tensor_Dense<vector<double>, double>::operator=(
    const view_tensor_Dense<matrix::Dense<double>, double> &tens);
template void view_tensor_Dense<matrix::Dense<double>, double>::operator=(
    const view_tensor_Dense<matrix::Dense<double>, double> &tens);
template void
view_tensor_Dense<tensor::tensor_Dense<double>, double>::operator=(
    const view_tensor_Dense<matrix::Dense<double>, double> &tens);
template void view_tensor_Dense<vector<float>, float>::operator=(
    const view_tensor_Dense<matrix::Dense<float>, float> &tens);
template void view_tensor_Dense<matrix::Dense<float>, float>::operator=(
    const view_tensor_Dense<matrix::Dense<float>, float> &tens);
template void view_tensor_Dense<tensor::tensor_Dense<float>, float>::operator=(
    const view_tensor_Dense<matrix::Dense<float>, float> &tens);

template <typename T, typename U>
void view_tensor_Dense<T, U>::operator=(
    const view_tensor_Dense<tensor::tensor_Dense<U>, U> &tens) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  // err
  assert(monolish::util::is_same_size(*this, tens));
  assert(monolish::util::is_same_device_mem_stat(*this, tens));

  // gpu copy
  if (tens.get_device_mem_stat()) {
    internal::vcopy(get_nnz(), tens.begin(), begin(), true);
  } else {
    internal::vcopy(get_nnz(), tens.begin(), begin(), false);
  }

  logger.util_out();
}

template void view_tensor_Dense<vector<double>, double>::operator=(
    const view_tensor_Dense<tensor::tensor_Dense<double>, double> &tens);
template void view_tensor_Dense<matrix::Dense<double>, double>::operator=(
    const view_tensor_Dense<tensor::tensor_Dense<double>, double> &tens);
template void
view_tensor_Dense<tensor::tensor_Dense<double>, double>::operator=(
    const view_tensor_Dense<tensor::tensor_Dense<double>, double> &tens);
template void view_tensor_Dense<vector<float>, float>::operator=(
    const view_tensor_Dense<tensor::tensor_Dense<float>, float> &tens);
template void view_tensor_Dense<matrix::Dense<float>, float>::operator=(
    const view_tensor_Dense<tensor::tensor_Dense<float>, float> &tens);
template void view_tensor_Dense<tensor::tensor_Dense<float>, float>::operator=(
    const view_tensor_Dense<tensor::tensor_Dense<float>, float> &tens);

} // namespace monolish
