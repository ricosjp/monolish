#include "../../../include/monolish/common/monolish_common.hpp"
#include "../../../include/monolish/common/monolish_dense.hpp"
#include "../../../include/monolish/common/monolish_logger.hpp"
#include "../../../include/monolish/common/monolish_matrix.hpp"
#include "../../../include/monolish/common/monolish_tensor.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {
namespace tensor {

template <typename T>
void tensor_Dense<T>::operator=(const tensor_Dense<T> &tens) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  // err
  assert(monolish::util::is_same_size(*this, tens));
  assert(monolish::util::is_same_device_mem_stat(*this, tens));
  val_create_flag = true;

  // gpu copy
  internal::vcopy(get_nnz(), tens.begin(), begin(), get_device_mem_stat());

  logger.util_out();
}

template void tensor_Dense<double>::operator=(const tensor_Dense<double> &tens);
template void tensor_Dense<float>::operator=(const tensor_Dense<float> &tens);

template <typename T>
void tensor_Dense<T>::set_ptr(const std::vector<size_t> &shape,
                              const T *value) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  val_create_flag = true;
  resize(shape);

  internal::vcopy(get_nnz(), value, begin(), false);

  logger.util_out();
}
template void tensor_Dense<double>::set_ptr(const std::vector<size_t> &shape,
                                            const double *value);
template void tensor_Dense<float>::set_ptr(const std::vector<size_t> &shape,
                                           const float *value);

template <typename T>
void tensor_Dense<T>::set_ptr(const std::vector<size_t> &shape, const T value) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  val_create_flag = true;
  resize(shape);

  internal::vbroadcast(get_nnz(), value, begin(), false);

  logger.util_out();
}
template void tensor_Dense<double>::set_ptr(const std::vector<size_t> &shape,
                                            const double value);
template void tensor_Dense<float>::set_ptr(const std::vector<size_t> &shape,
                                           const float value);

template <typename T>
void tensor_Dense<T>::set_ptr(const std::vector<size_t> &shape,
                              const std::vector<T> &value) {
  set_ptr(shape, value.data());
}
template void tensor_Dense<double>::set_ptr(const std::vector<size_t> &shape,
                                            const std::vector<double> &value);
template void tensor_Dense<float>::set_ptr(const std::vector<size_t> &shape,
                                           const std::vector<float> &value);

} // namespace tensor
} // namespace monolish
