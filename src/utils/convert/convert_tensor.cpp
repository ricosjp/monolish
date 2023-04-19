#include "../../../include/monolish/common/monolish_dense.hpp"
#include "../../../include/monolish/common/monolish_logger.hpp"
#include "../../../include/monolish/common/monolish_matrix.hpp"
#include "../../../include/monolish/common/monolish_tensor.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {
namespace tensor {

template <typename T>
void tensor_Dense<T>::convert(const matrix::Dense<T> &mat) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  vad_create_flag = true;
  this->shape.resize(2);
  this->shape[0] = mat.get_row();
  this->shape[1] = mat.get_col();
  resize(mat.get_nnz());

#if MONOLISH_USE_NVIDIA_GPU
  if (mat.get_device_mem_stat()) {
    throw std::runtime_error(
        "error can not convert CRS->CRS when gpu_status == true");
  }
#endif
  internal::vcopy(get_nnz(), mat.data(), data(), false);

  logger.util_out();
}
template void tensor_Dense<double>::convert(const matrix::Dense<double> &mat);
template void tensor_Dense<float>::convert(const matrix::Dense<float> &mat);

template <typename T> void tensor_Dense<T>::convert(const vector<T> &vec) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  vad_create_flag = true;
  this->shape.resize(1);
  this->shape[0] = vec.get_nnz();
  resize(vec.get_nnz());

#if MONOLISH_USE_NVIDIA_GPU
  if (mat.get_device_mem_stat()) {
    throw std::runtime_error(
        "error can not convert CRS->CRS when gpu_status == true");
  }
#endif
  internal::vcopy(get_nnz(), vec.data(), data(), false);

  logger.util_out();
}
template void tensor_Dense<double>::convert(const vector<double> &vec);
template void tensor_Dense<float>::convert(const vector<float> &vec);

} // namespace tensor
} // namespace monolish
