#include "../../include/monolish/common/monolish_dense.hpp"
#include "../../include/monolish/common/monolish_logger.hpp"
#include "../../include/monolish/common/monolish_matrix.hpp"
#include "../../include/monolish/common/monolish_vector.hpp"
#include "../internal/monolish_internal.hpp"

namespace monolish {
namespace matrix {

// matrix constructor ///

template <typename T> Dense<T>::Dense(const size_t M, const size_t N) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  set_row(M);
  set_col(N);

  vad_create_flag = true;
  resize(M*N);
  logger.util_out();
}
template Dense<double>::Dense(const size_t M, const size_t N);
template Dense<float>::Dense(const size_t M, const size_t N);

template <typename T>
Dense<T>::Dense(const size_t M, const size_t N, const T *value) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  set_row(M);
  set_col(N);

  vad_create_flag = true;
  resize(M*N);
  std::copy(value, value + get_nnz(), data());
  logger.util_out();
}
template Dense<double>::Dense(const size_t M, const size_t N,
                              const double *value);
template Dense<float>::Dense(const size_t M, const size_t N,
                             const float *value);

template <typename T>
Dense<T>::Dense(const size_t M, const size_t N, const std::vector<T> &value) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  set_row(M);
  set_col(N);

  vad_create_flag = true;
  resize(M*N);
  std::copy(value.begin(), value.end(), data());
  logger.util_out();
}
template Dense<double>::Dense(const size_t M, const size_t N,
                              const std::vector<double> &value);
template Dense<float>::Dense(const size_t M, const size_t N,
                             const std::vector<float> &value);

template <typename T>
Dense<T>::Dense(const size_t M, const size_t N, const vector<T> &value) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  set_row(M);
  set_col(N);

  vad_create_flag = true;
  resize(M*N);
  std::copy(value.data(), value.data() + get_nnz(), data());

  if (value.get_device_mem_stat() == true) {
#if MONOLISH_USE_NVIDIA_GPU
    send();
    T *vald = data();
    const T *data = value.data();
#pragma omp target teams distribute parallel for
    for (size_t i = 0; i < get_nnz(); i++) {
      vald[i] = data[i];
    }
#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  }

  logger.util_out();
}
template Dense<double>::Dense(const size_t M, const size_t N,
                              const vector<double> &value);
template Dense<float>::Dense(const size_t M, const size_t N,
                             const vector<float> &value);

template <typename T>
Dense<T>::Dense(const size_t M, const size_t N,
                const std::initializer_list<T> &list) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  set_row(M);
  set_col(N);

  vad_create_flag = true;
  resize(M*N);
  std::copy(list.begin(), list.end(), data());
  logger.util_out();
}
template Dense<double>::Dense(const size_t M, const size_t N,
                              const std::initializer_list<double> &list);
template Dense<float>::Dense(const size_t M, const size_t N,
                             const std::initializer_list<float> &list);

template <typename T>
Dense<T>::Dense(const size_t M, const size_t N, const T min, const T max) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  set_row(M);
  set_col(N);

  vad_create_flag = true;
  resize(M*N);

  std::random_device random;
  std::mt19937 mt(random());
  std::uniform_real_distribution<> rand(min, max);

  for (size_t i = 0; i < get_nnz(); i++) {
    data()[i] = rand(mt);
  }

  logger.util_out();
}
template Dense<double>::Dense(const size_t M, const size_t N, const double min,
                              const double max);
template Dense<float>::Dense(const size_t M, const size_t N, const float min,
                             const float max);

template <typename T>
Dense<T>::Dense(const size_t M, const size_t N, const T min, const T max,
                const std::uint32_t seed) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  set_row(M);
  set_col(N);

  vad_create_flag = true;
  resize(M*N);

  std::mt19937 mt(seed);
  std::uniform_real_distribution<> rand(min, max);

  for (size_t i = 0; i < get_nnz(); i++) {
    data()[i] = rand(mt);
  }

  logger.util_out();
}
template Dense<double>::Dense(const size_t M, const size_t N, const double min,
                              const double max, const std::uint32_t seed);
template Dense<float>::Dense(const size_t M, const size_t N, const float min,
                             const float max, const std::uint32_t seed);

template <typename T>
Dense<T>::Dense(const size_t M, const size_t N, const T value) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  set_row(M);
  set_col(N);

  vad_create_flag = true;
  resize(M*N);

#pragma omp parallel for
  for (size_t i = 0; i < get_nnz(); i++) {
    data()[i] = value;
  }

  logger.util_out();
}
template Dense<double>::Dense(const size_t M, const size_t N,
                              const double value);
template Dense<float>::Dense(const size_t M, const size_t N, const float value);

// copy constructor///////////////////////////////
template <typename T> Dense<T>::Dense(const Dense<T> &mat) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  vad_create_flag = true;
  resize(mat.get_nnz());

  rowN = mat.get_row();
  colN = mat.get_col();

#if MONOLISH_USE_NVIDIA_GPU
  if (mat.get_device_mem_stat()) {
    send();
    internal::vcopy(get_nnz(), mat.data(), data(), true);
  }
#endif
  internal::vcopy(get_nnz(), mat.data(), data(), false);

  logger.util_out();
}
template Dense<double>::Dense(const Dense<double> &mat);
template Dense<float>::Dense(const Dense<float> &mat);

// initialization constructor///////////////////////////////
template <typename T> Dense<T>::Dense(const Dense<T> &mat, T value) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  vad_create_flag = true;
  resize(mat.get_nnz());

  rowN = mat.get_row();
  colN = mat.get_col();

#if MONOLISH_USE_NVIDIA_GPU
  if (mat.get_device_mem_stat()) {
    send();
    internal::vbroadcast(get_nnz(), value, data(), true);
  }
#endif
  internal::vbroadcast(get_nnz(), value, data(), false);

  logger.util_out();
}
template Dense<double>::Dense(const Dense<double> &mat, double value);
template Dense<float>::Dense(const Dense<float> &mat, float value);

} // namespace matrix
} // namespace monolish
