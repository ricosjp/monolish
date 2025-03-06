#include "../../../include/monolish/common/monolish_dense.hpp"
#include "../../../include/monolish/common/monolish_logger.hpp"
#include "../../../include/monolish/common/monolish_tensor.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {
namespace tensor {

template <typename T> void tensor_CRS<T>::fill(T value) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  if (get_device_mem_stat() == true) {
#if MONOLISH_USE_NVIDIA_GPU
    T *vald = data();
#pragma omp target teams distribute parallel for
    for (auto i = decltype(get_nnz()){0}; i < get_nnz(); i++) {
      vald[i] = value;
    }
#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  } else {
#pragma omp parallel for
    for (auto i = decltype(get_nnz()){0}; i < get_nnz(); i++) {
      data()[i] = value;
    }
  }
  logger.util_out();
}
template void tensor_CRS<double>::fill(double value);
template void tensor_CRS<float>::fill(float value);

} // namespace tensor
} // namespace monolish
