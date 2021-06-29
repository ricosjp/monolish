#include "../../../include/common/monolish_dense.hpp"
#include "../../../include/common/monolish_logger.hpp"
#include "../../../include/common/monolish_matrix.hpp"
#include "../../../include/monolish_vml.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {

template <typename T> void vector<T>::fill(T value) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  if (get_device_mem_stat() == true) {
#if MONOLISH_USE_NVIDIA_GPU
    T *vald = val.data();
    auto N = val.size();
#pragma omp target teams distribute parallel for
    for (size_t i = 0; i < N; i++) {
      vald[i] = value;
    }
#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  } else {
#pragma omp parallel for
    for (size_t i = 0; i < val.size(); i++) {
      val[i] = value;
    }
  }
  logger.util_out();
}
template void vector<double>::fill(double value);
template void vector<float>::fill(float value);

} // namespace monolish
