#include "../../include/common/monolish_dense.hpp"
#include "../../include/common/monolish_logger.hpp"
#include "../../include/common/monolish_matrix.hpp"
#include "../../include/common/monolish_vector.hpp"
#include "../internal/monolish_internal.hpp"

namespace monolish {
namespace matrix {

// matrix constructor ///

template <typename T> Dense<T>::Dense(const size_t M, const size_t N) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  set_row(M);
  set_col(N);
  set_nnz(N * M);

  val.resize(nnz);
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
  set_nnz(M * N);

  val.resize(nnz);
  std::copy(value, value + nnz, val.begin());
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
  set_nnz(M * N);

  val.resize(nnz);
  std::copy(value.begin(), value.end(), val.begin());
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
  set_nnz(M * N);

  val.resize(nnz);
  std::copy(value.data(), value.data() + nnz, val.begin());

  if (value.get_device_mem_stat() == true) {
#if MONOLISH_USE_NVIDIA_GPU
    send();
    const T *data = value.data();
    T *vald = val.data();
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
  set_nnz(M * N);

  val.resize(nnz);
  std::copy(list.begin(), list.end(), val.begin());
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
  set_nnz(M * N);

  val.resize(nnz);

#pragma omp parallel
  {
    std::random_device random;
    std::mt19937 mt(random());
    std::uniform_real_distribution<> rand(min, max);

#pragma omp for
    for (size_t i = 0; i < val.size(); i++) {
      val[i] = rand(mt);
    }
  }
  logger.util_out();
}
template Dense<double>::Dense(const size_t M, const size_t N, const double min,
                              const double max);
template Dense<float>::Dense(const size_t M, const size_t N, const float min,
                             const float max);

template <typename T>
Dense<T>::Dense(const size_t M, const size_t N, const T value) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  set_row(M);
  set_col(N);
  set_nnz(M * N);

  val.resize(nnz);

#pragma omp parallel for
  for (size_t i = 0; i < val.size(); i++) {
    val[i] = value;
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

  val.resize(mat.get_nnz());

  rowN = mat.get_row();
  colN = mat.get_col();
  nnz = mat.get_nnz();

#if MONOLISH_USE_NVIDIA_GPU
  if (mat.get_device_mem_stat()) {
    send();
    internal::vcopy(get_nnz(), mat.val.data(), val.data(), true);
  }
#endif
  internal::vcopy(get_nnz(), mat.val.data(), val.data(), false);

  logger.util_out();
}
template Dense<double>::Dense(const Dense<double> &mat);
template Dense<float>::Dense(const Dense<float> &mat);

} // namespace matrix
} // namespace monolish
