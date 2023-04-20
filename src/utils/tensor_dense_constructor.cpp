#include "../../include/monolish/common/monolish_dense.hpp"
#include "../../include/monolish/common/monolish_logger.hpp"
#include "../../include/monolish/common/monolish_matrix.hpp"
#include "../../include/monolish/common/monolish_tensor_dense.hpp"
#include "../../include/monolish/common/monolish_vector.hpp"
#include "../internal/monolish_internal.hpp"

namespace monolish {
namespace tensor {

template <typename T>
tensor_Dense<T>::tensor_Dense(const std::vector<size_t> &shape) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  this->shape = shape;

  vad_create_flag = true;
  resize(shape);
  logger.util_out();
}
template tensor_Dense<double>::tensor_Dense(const std::vector<size_t> &shape);
template tensor_Dense<float>::tensor_Dense(const std::vector<size_t> &shape);

template <typename T>
tensor_Dense<T>::tensor_Dense(const std::vector<size_t> &shape,
                              const T *value) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  this->shape = shape;

  vad_create_flag = true;
  resize(shape);
  std::copy(value, value + get_nnz(), data());
  logger.util_out();
}

template tensor_Dense<double>::tensor_Dense(const std::vector<size_t> &shape,
                                            const double *value);
template tensor_Dense<float>::tensor_Dense(const std::vector<size_t> &shape,
                                           const float *value);

template <typename T>
tensor_Dense<T>::tensor_Dense(const std::vector<size_t> &shape, const T min,
                              const T max) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  this->shape = shape;

  vad_create_flag = true;
  resize(shape);

  std::random_device random;
  std::mt19937 mt(random());
  std::uniform_real_distribution<> rand(min, max);

  for (size_t i = 0; i < get_nnz(); i++) {
    data()[i] = rand(mt);
  }

  logger.util_out();
}
template tensor_Dense<double>::tensor_Dense(const std::vector<size_t> &shape,
                                            const double min, const double max);
template tensor_Dense<float>::tensor_Dense(const std::vector<size_t> &shape,
                                           const float min, const float max);

template <typename T>
tensor_Dense<T>::tensor_Dense(const std::vector<size_t> &shape, const T min,
                              const T max, const std::uint32_t seed) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  this->shape = shape;

  vad_create_flag = true;
  resize(shape);

  std::mt19937 mt(seed);
  std::uniform_real_distribution<> rand(min, max);

  for (size_t i = 0; i < get_nnz(); i++) {
    data()[i] = rand(mt);
  }

  logger.util_out();
}
template tensor_Dense<double>::tensor_Dense(const std::vector<size_t> &shape,
                                            const double min, const double max,
                                            const std::uint32_t seed);
template tensor_Dense<float>::tensor_Dense(const std::vector<size_t> &shape,
                                           const float min, const float max,
                                           const std::uint32_t seed);

template <typename T>
tensor_Dense<T>::tensor_Dense(const tensor_Dense<T> &tens) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  vad_create_flag = true;
  resize(tens.get_nnz());
  this->shape = tens.get_shape();

#if MONOLISH_USE_NVIDIA_GPU
  if (tens.get_device_mem_stat()) {
    send();
    internal::vcopy(get_nnz(), tens.data(), data(), true);
  }
#endif
  internal::vcopy(get_nnz(), tens.data(), data(), false);

  logger.util_out();
}
template tensor_Dense<double>::tensor_Dense(const tensor_Dense<double> &tens);
template tensor_Dense<float>::tensor_Dense(const tensor_Dense<float> &tens);

template <typename T>
tensor_Dense<T>::tensor_Dense(const tensor_Dense<T> &tens, T value) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  vad_create_flag = true;
  resize(tens.get_nnz());
  this->shape = tens.get_shape();

#if MONOLISH_USE_NVIDIA_GPU
  if (tens.get_device_mem_stat()) {
    send();
    internal::vbroadcast(get_nnz(), value, data(), true);
  }
#endif
  internal::vbroadcast(get_nnz(), value, data(), false);

  logger.util_out();
}

} // namespace tensor
} // namespace monolish
