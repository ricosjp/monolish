#include "../../include/monolish_blas.hpp"
#include "../../include/monolish_vml.hpp"
#include "../internal/monolish_internal.hpp"

namespace monolish {

/////constructor//////
template <typename T> vector<T>::vector(const size_t N) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  val_create_flag = true;
  resize(N);
  logger.util_out();
}
template vector<double>::vector(const size_t N);
template vector<float>::vector(const size_t N);

template <typename T> vector<T>::vector(const size_t N, const T value) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  val_create_flag = true;
  resize(N);

#pragma omp parallel for
  for (size_t i = 0; i < N; i++) {
    data()[i] = value;
  }

  logger.util_out();
}
template vector<double>::vector(const size_t N, const double value);
template vector<float>::vector(const size_t N, const float value);

// random
template <typename T>
void util::random_vector(vector<T> &vec, const T min, const T max) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  std::random_device random;
  std::mt19937 mt(random());
  std::uniform_real_distribution<> rand(min, max);

  for (size_t i = 0; i < vec.size(); i++) {
    vec[i] = rand(mt);
  }

  logger.util_out();
}
template void util::random_vector(vector<double> &vec, const double min,
                                  const double max);
template void util::random_vector(vector<float> &vec, const float min,
                                  const float max);

template <typename T>
vector<T>::vector(const size_t N, const T min, const T max) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  val_create_flag = true;
  resize(N);

  std::random_device random;
  std::mt19937 mt(random());
  std::uniform_real_distribution<> rand(min, max);

  for (size_t i = 0; i < N; i++) {
    data()[i] = rand(mt);
  }

  logger.util_out();
}
template vector<double>::vector(const size_t N, const double min,
                                const double max);
template vector<float>::vector(const size_t N, const float min,
                               const float max);

// random_seed
template <typename T>
void util::random_vector(vector<T> &vec, const T min, const T max,
                         const std::uint32_t seed) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  std::mt19937 mt(seed);
  std::uniform_real_distribution<> rand(min, max);

  for (size_t i = 0; i < vec.size(); i++) {
    vec[i] = rand(mt);
  }

  logger.util_out();
}
template void util::random_vector(vector<double> &vec, const double min,
                                  const double max, const std::uint32_t seed);
template void util::random_vector(vector<float> &vec, const float min,
                                  const float max, const std::uint32_t seed);

template <typename T>
vector<T>::vector(const size_t N, const T min, const T max,
                  const std::uint32_t seed) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  val_create_flag = true;
  resize(N);

  std::mt19937 mt(seed);
  std::uniform_real_distribution<> rand(min, max);

  for (size_t i = 0; i < N; i++) {
    data()[i] = rand(mt);
  }

  logger.util_out();
}
template vector<double>::vector(const size_t N, const double min,
                                const double max, const std::uint32_t seed);
template vector<float>::vector(const size_t N, const float min, const float max,
                               const std::uint32_t seed);

// start-end
template <typename T> vector<T>::vector(const T *start, const T *end) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  size_t size = (end - start);
  val_create_flag = true;
  resize(size);
  std::copy(start, end, data());
  logger.util_out();
}
template vector<double>::vector(const double *start, const double *end);
template vector<float>::vector(const float *start, const float *end);

// copy constructor
template <typename T> vector<T>::vector(const std::vector<T> &vec) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  val_create_flag = true;
  resize(vec.size());
  std::copy(vec.begin(), vec.end(), data());
  logger.util_out();
}

template vector<double>::vector(const std::vector<double> &vec);
template vector<float>::vector(const std::vector<float> &vec);

// copy constructor
template <typename T> vector<T>::vector(const std::initializer_list<T> &list) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  val_create_flag = true;
  resize(list.size());
  std::copy(list.begin(), list.end(), data());
  logger.util_out();
}

template vector<double>::vector(const std::initializer_list<double> &list);
template vector<float>::vector(const std::initializer_list<float> &list);

template <typename T> vector<T>::vector(const monolish::vector<T> &vec) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  val_create_flag = true;
  resize(vec.size());

  // gpu copy and recv
#if MONOLISH_USE_NVIDIA_GPU
  if (vec.get_device_mem_stat()) {
    send();
    internal::vcopy(vec.size(), vec.data(), data(), true);
  }
#endif
  internal::vcopy(vec.size(), vec.data(), data(), false);

  logger.util_out();
}

template vector<double>::vector(const vector<double> &vec);
template vector<float>::vector(const vector<float> &vec);

template <typename T>
vector<T>::vector(const monolish::view1D<monolish::vector<T>, T> &vec) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  val_create_flag = true;
  resize(vec.size());

  // gpu copy and recv
#if MONOLISH_USE_NVIDIA_GPU
  if (vec.get_device_mem_stat()) {
    send();
    internal::vcopy(vec.size(), vec.data() + vec.get_offset(), data(), true);
  }
#endif
  internal::vcopy(vec.size(), vec.data() + vec.get_offset(), data(), false);

  logger.util_out();
}

template vector<double>::vector(const view1D<vector<double>, double> &vec);
template vector<float>::vector(const view1D<vector<float>, float> &vec);

template <typename T>
vector<T>::vector(const monolish::view1D<monolish::matrix::Dense<T>, T> &vec) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  val_create_flag = true;
  resize(vec.size());

  // gpu copy and recv
#if MONOLISH_USE_NVIDIA_GPU
  if (vec.get_device_mem_stat()) {
    send();
    internal::vcopy(vec.size(), vec.data() + vec.get_offset(), data(), true);
  }
#endif
  internal::vcopy(vec.size(), vec.data() + vec.get_offset(), data(), false);

  logger.util_out();
}

template vector<double>::vector(
    const view1D<matrix::Dense<double>, double> &vec);
template vector<float>::vector(const view1D<matrix::Dense<float>, float> &vec);

template <typename T>
vector<T>::vector(
    const monolish::view1D<monolish::tensor::tensor_Dense<T>, T> &vec) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  val_create_flag = true;
  resize(vec.size());

  // gpu copy and recv
#if MONOLISH_USE_NVIDIA_GPU
  if (vec.get_device_mem_stat()) {
    send();
    internal::vcopy(vec.size(), vec.data() + vec.get_offset(), data(), true);
  }
#endif
  internal::vcopy(vec.size(), vec.data() + vec.get_offset(), data(), false);

  logger.util_out();
}

template vector<double>::vector(
    const view1D<tensor::tensor_Dense<double>, double> &vec);
template vector<float>::vector(
    const view1D<tensor::tensor_Dense<float>, float> &vec);

} // namespace monolish
