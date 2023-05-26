#include "../../../include/monolish/common/monolish_common.hpp"
#include "../../../include/monolish/common/monolish_dense.hpp"
#include "../../../include/monolish/common/monolish_logger.hpp"
#include "../../../include/monolish/common/monolish_matrix.hpp"
#include "../../../include/monolish_vml.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {

template <typename T, typename U>
void view1D<T, U>::operator=(const std::vector<U> &vec) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  resize(vec.size());
  std::copy(vec.begin(), vec.end(), begin());

  logger.util_out();
}

template void
view1D<vector<double>, double>::operator=(const std::vector<double> &vec);
template void view1D<matrix::Dense<double>, double>::operator=(
    const std::vector<double> &vec);
template void view1D<tensor::tensor_Dense<double>, double>::operator=(
    const std::vector<double> &vec);
template void
view1D<vector<float>, float>::operator=(const std::vector<float> &vec);
template void
view1D<matrix::Dense<float>, float>::operator=(const std::vector<float> &vec);
template void view1D<tensor::tensor_Dense<float>, float>::operator=(
    const std::vector<float> &vec);

template <typename T, typename U>
void view1D<T, U>::operator=(const vector<U> &vec) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  // err
  assert(monolish::util::is_same_size(*this, vec));
  assert(monolish::util::is_same_device_mem_stat(*this, vec));

  // gpu copy and recv
  if (vec.get_device_mem_stat()) {
#if MONOLISH_USE_NVIDIA_GPU
    internal::vcopy(vec.size(), vec.begin(), begin(), true);
#endif
  } else {
    internal::vcopy(vec.size(), vec.begin(), begin(), false);
  }

  logger.util_out();
}

template void
view1D<vector<double>, double>::operator=(const vector<double> &vec);
template void
view1D<matrix::Dense<double>, double>::operator=(const vector<double> &vec);
template void view1D<tensor::tensor_Dense<double>, double>::operator=(
    const vector<double> &vec);
template void view1D<vector<float>, float>::operator=(const vector<float> &vec);
template void
view1D<matrix::Dense<float>, float>::operator=(const vector<float> &vec);
template void
view1D<tensor::tensor_Dense<float>, float>::operator=(const vector<float> &vec);

template <typename T, typename U>
void view1D<T, U>::operator=(const view1D<vector<U>, U> &vec) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  // err
  assert(monolish::util::is_same_size(*this, vec));
  assert(monolish::util::is_same_device_mem_stat(*this, vec));

  // gpu copy and recv
  if (vec.get_device_mem_stat()) {
#if MONOLISH_USE_NVIDIA_GPU
    internal::vcopy(vec.size(), vec.begin() + vec.get_offset(), begin(), true);
#endif
  } else {
    internal::vcopy(vec.size(), vec.begin() + vec.get_offset(), begin(), false);
  }

  logger.util_out();
}
template void view1D<vector<double>, double>::operator=(
    const view1D<vector<double>, double> &vec);
template void view1D<matrix::Dense<double>, double>::operator=(
    const view1D<vector<double>, double> &vec);
template void view1D<tensor::tensor_Dense<double>, double>::operator=(
    const view1D<vector<double>, double> &vec);
template void view1D<vector<float>, float>::operator=(
    const view1D<vector<float>, float> &vec);
template void view1D<matrix::Dense<float>, float>::operator=(
    const view1D<vector<float>, float> &vec);
template void view1D<tensor::tensor_Dense<float>, float>::operator=(
    const view1D<vector<float>, float> &vec);

template <typename T, typename U>
void view1D<T, U>::operator=(const view1D<matrix::Dense<U>, U> &vec) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  // err
  assert(monolish::util::is_same_size(*this, vec));
  assert(monolish::util::is_same_device_mem_stat(*this, vec));

  // gpu copy and recv
  if (vec.get_device_mem_stat()) {
#if MONOLISH_USE_NVIDIA_GPU
    internal::vcopy(vec.size(), vec.begin() + vec.get_offset(), begin(), true);
#endif
  } else {
    internal::vcopy(vec.size(), vec.begin() + vec.get_offset(), begin(), false);
  }

  logger.util_out();
}
template void view1D<vector<double>, double>::operator=(
    const view1D<matrix::Dense<double>, double> &vec);
template void view1D<matrix::Dense<double>, double>::operator=(
    const view1D<matrix::Dense<double>, double> &vec);
template void view1D<tensor::tensor_Dense<double>, double>::operator=(
    const view1D<matrix::Dense<double>, double> &vec);
template void view1D<vector<float>, float>::operator=(
    const view1D<matrix::Dense<float>, float> &vec);
template void view1D<matrix::Dense<float>, float>::operator=(
    const view1D<matrix::Dense<float>, float> &vec);
template void view1D<tensor::tensor_Dense<float>, float>::operator=(
    const view1D<matrix::Dense<float>, float> &vec);

template <typename T, typename U>
void view1D<T, U>::operator=(const view1D<tensor::tensor_Dense<U>, U> &vec) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  // err
  assert(monolish::util::is_same_size(*this, vec));
  assert(monolish::util::is_same_device_mem_stat(*this, vec));

  // gpu copy and recv
  if (vec.get_device_mem_stat()) {
#if MONOLISH_USE_NVIDIA_GPU
    internal::vcopy(vec.size(), vec.begin() + vec.get_offset(), begin(), true);
#endif
  } else {
    internal::vcopy(vec.size(), vec.begin() + vec.get_offset(), begin(), false);
  }

  logger.util_out();
}
template void view1D<vector<double>, double>::operator=(
    const view1D<tensor::tensor_Dense<double>, double> &vec);
template void view1D<matrix::Dense<double>, double>::operator=(
    const view1D<tensor::tensor_Dense<double>, double> &vec);
template void view1D<tensor::tensor_Dense<double>, double>::operator=(
    const view1D<tensor::tensor_Dense<double>, double> &vec);
template void view1D<vector<float>, float>::operator=(
    const view1D<tensor::tensor_Dense<float>, float> &vec);
template void view1D<matrix::Dense<float>, float>::operator=(
    const view1D<tensor::tensor_Dense<float>, float> &vec);
template void view1D<tensor::tensor_Dense<float>, float>::operator=(
    const view1D<tensor::tensor_Dense<float>, float> &vec);

} // namespace monolish
