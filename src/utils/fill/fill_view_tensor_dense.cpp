#include "../../../include/monolish/common/monolish_dense.hpp"
#include "../../../include/monolish/common/monolish_logger.hpp"
#include "../../../include/monolish/common/monolish_matrix.hpp"
#include "../../../include/monolish/common/monolish_tensor.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {

template <typename T, typename U> void view_tensor_Dense<T, U>::fill(U value) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  U *val = data();
  if (get_device_mem_stat() == true) {
#if MONOLISH_USE_NVIDIA_GPU
#pragma omp target teams distribute parallel for
    for (auto i = first; i < last; i++) {
      val[i] = value;
    }
#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  } else {
#pragma omp parallel for
    for (auto i = first; i < last; i++) {
      val[i] = value;
    }
  }
  logger.util_out();
}
template void view_tensor_Dense<vector<double>, double>::fill(double value);
template void
view_tensor_Dense<matrix::Dense<double>, double>::fill(double value);
// template void view_tensor_Dense<matrix::LinearOperator<double>,
// double>::fill(double value);
template void
view_tensor_Dense<tensor::tensor_Dense<double>, double>::fill(double value);
template void view_tensor_Dense<vector<float>, float>::fill(float value);
template void view_tensor_Dense<matrix::Dense<float>, float>::fill(float value);
// template void view_tensor_Dense<matrix::LinearOperator<float>,
// float>::fill(float value);
template void
view_tensor_Dense<tensor::tensor_Dense<float>, float>::fill(float value);

} // namespace monolish
