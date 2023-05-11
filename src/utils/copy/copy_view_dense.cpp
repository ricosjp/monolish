#include "../../../include/monolish/common/monolish_common.hpp"
#include "../../../include/monolish/common/monolish_dense.hpp"
#include "../../../include/monolish/common/monolish_logger.hpp"
#include "../../../include/monolish/common/monolish_matrix.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {

template <typename T, typename U>
void view_Dense<T, U>::operator=(const matrix::Dense<U> &mat) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  // err
  assert(monolish::util::is_same_size(*this, mat));
  assert(monolish::util::is_same_device_mem_stat(*this, mat));

  // gpu copy
  if (mat.get_device_mem_stat()) {
    internal::vcopy(get_nnz(), mat.begin(), begin(), true);
  } else {
    internal::vcopy(get_nnz(), mat.begin(), begin(), false);
  }

  logger.util_out();
}

template void
view_Dense<vector<double>, double>::operator=(const matrix::Dense<double> &mat);
template void view_Dense<matrix::Dense<double>, double>::operator=(
    const matrix::Dense<double> &mat);
template void view_Dense<tensor::tensor_Dense<double>, double>::operator=(
    const matrix::Dense<double> &mat);
template void
view_Dense<vector<float>, float>::operator=(const matrix::Dense<float> &mat);
template void view_Dense<matrix::Dense<float>, float>::operator=(
    const matrix::Dense<float> &mat);
template void view_Dense<tensor::tensor_Dense<float>, float>::operator=(
    const matrix::Dense<float> &mat);

template <typename T, typename U>
void view_Dense<T, U>::operator=(const view_Dense<vector<U>, U> &mat) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  // err
  assert(monolish::util::is_same_size(*this, mat));
  assert(monolish::util::is_same_device_mem_stat(*this, mat));

  // gpu copy
  if (mat.get_device_mem_stat()) {
    internal::vcopy(get_nnz(), mat.begin(), begin(), true);
  } else {
    internal::vcopy(get_nnz(), mat.begin(), begin(), false);
  }

  logger.util_out();
}

template void view_Dense<vector<double>, double>::operator=(
    const view_Dense<vector<double>, double> &mat);
template void view_Dense<matrix::Dense<double>, double>::operator=(
    const view_Dense<vector<double>, double> &mat);
template void view_Dense<tensor::tensor_Dense<double>, double>::operator=(
    const view_Dense<vector<double>, double> &mat);
template void view_Dense<vector<float>, float>::operator=(
    const view_Dense<vector<float>, float> &mat);
template void view_Dense<matrix::Dense<float>, float>::operator=(
    const view_Dense<vector<float>, float> &mat);
template void view_Dense<tensor::tensor_Dense<float>, float>::operator=(
    const view_Dense<vector<float>, float> &mat);

template <typename T, typename U>
void view_Dense<T, U>::operator=(const view_Dense<matrix::Dense<U>, U> &mat) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  // err
  assert(monolish::util::is_same_size(*this, mat));
  assert(monolish::util::is_same_device_mem_stat(*this, mat));

  // gpu copy
  if (mat.get_device_mem_stat()) {
    internal::vcopy(get_nnz(), mat.begin(), begin(), true);
  } else {
    internal::vcopy(get_nnz(), mat.begin(), begin(), false);
  }

  logger.util_out();
}

template void view_Dense<vector<double>, double>::operator=(
    const view_Dense<matrix::Dense<double>, double> &mat);
template void view_Dense<matrix::Dense<double>, double>::operator=(
    const view_Dense<matrix::Dense<double>, double> &mat);
template void view_Dense<tensor::tensor_Dense<double>, double>::operator=(
    const view_Dense<matrix::Dense<double>, double> &mat);
template void view_Dense<vector<float>, float>::operator=(
    const view_Dense<matrix::Dense<float>, float> &mat);
template void view_Dense<matrix::Dense<float>, float>::operator=(
    const view_Dense<matrix::Dense<float>, float> &mat);
template void view_Dense<tensor::tensor_Dense<float>, float>::operator=(
    const view_Dense<matrix::Dense<float>, float> &mat);

template <typename T, typename U>
void view_Dense<T, U>::operator=(
    const view_Dense<tensor::tensor_Dense<U>, U> &mat) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  // err
  assert(monolish::util::is_same_size(*this, mat));
  assert(monolish::util::is_same_device_mem_stat(*this, mat));

  // gpu copy
  if (mat.get_device_mem_stat()) {
    internal::vcopy(get_nnz(), mat.begin(), begin(), true);
  } else {
    internal::vcopy(get_nnz(), mat.begin(), begin(), false);
  }

  logger.util_out();
}

template void view_Dense<vector<double>, double>::operator=(
    const view_Dense<tensor::tensor_Dense<double>, double> &mat);
template void view_Dense<matrix::Dense<double>, double>::operator=(
    const view_Dense<tensor::tensor_Dense<double>, double> &mat);
template void view_Dense<tensor::tensor_Dense<double>, double>::operator=(
    const view_Dense<tensor::tensor_Dense<double>, double> &mat);
template void view_Dense<vector<float>, float>::operator=(
    const view_Dense<tensor::tensor_Dense<float>, float> &mat);
template void view_Dense<matrix::Dense<float>, float>::operator=(
    const view_Dense<tensor::tensor_Dense<float>, float> &mat);
template void view_Dense<tensor::tensor_Dense<float>, float>::operator=(
    const view_Dense<tensor::tensor_Dense<float>, float> &mat);

} // namespace monolish
