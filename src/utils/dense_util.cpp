#include "../../include/common/monolish_dense.hpp"
#include "../../include/common/monolish_logger.hpp"
#include "../../include/common/monolish_matrix.hpp"
#include "../../include/common/monolish_vector.hpp"
#include "../internal/monolish_internal.hpp"

#include <cassert>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>

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
Dense<T>::Dense(const size_t M, const size_t N, const std::vector<T> value) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  set_row(M);
  set_col(N);
  set_nnz(M * N);

  val.resize(nnz);
  std::copy(value.data(), value.data() + nnz, val.begin());
  logger.util_out();
}
template Dense<double>::Dense(const size_t M, const size_t N,
                              const std::vector<double> value);
template Dense<float>::Dense(const size_t M, const size_t N,
                             const std::vector<float> value);

template <typename T>
Dense<T>::Dense(const size_t M, const size_t N, const T min, const T max) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  set_row(M);
  set_col(N);
  set_nnz(M * N);

  val.resize(nnz);

  std::random_device random;
  std::mt19937 mt(random());
  std::uniform_real_distribution<> rand(min, max);

#pragma omp parallel for
  for (size_t i = 0; i < val.size(); i++) {
    val[i] = rand(mt);
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

#if MONOLISH_USE_GPU
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

// matrix utils ///

template <typename T> void Dense<T>::fill(T value) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  if (get_device_mem_stat() == true) {
#if MONOLISH_USE_GPU
#pragma omp target teams distribute parallel for
    for (size_t i = 0; i < get_nnz(); i++) {
      val[i] = value;
    }
#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  } else {
#pragma omp parallel for
    for (size_t i = 0; i < get_nnz(); i++) {
      val[i] = value;
    }
  }
  logger.util_out();
}
template void Dense<double>::fill(double value);
template void Dense<float>::fill(float value);

template <typename T> T Dense<T>::at(const size_t i, const size_t j) {
  if (get_device_mem_stat()) {
    throw std::runtime_error("at() Error, GPU vector cant use operator[]");
  }
  if (get_row() < i) {
    throw std::runtime_error("at() Error, A.row < i");
  }
  if (get_col() < j) {
    throw std::runtime_error("at() Error, A.col < j");
  }
  return val[get_col() * i + j];
}
template double Dense<double>::at(const size_t i, const size_t j);
template float Dense<float>::at(const size_t i, const size_t j);

template <typename T> T Dense<T>::at(const size_t i, const size_t j) const {
  if (get_device_mem_stat()) {
    throw std::runtime_error("at() Error, GPU vector cant use operator[]");
  }
  if (get_row() < i) {
    throw std::runtime_error("at() Error, A.row < i");
  }
  if (get_col() < j) {
    throw std::runtime_error("at() Error, A.col < j");
  }
  return val[get_col() * i + j];
}
template double Dense<double>::at(const size_t i, const size_t j) const;
template float Dense<float>::at(const size_t i, const size_t j) const;

template <typename T>
void Dense<T>::insert(const size_t i, const size_t j, const T Val) {
  if (get_device_mem_stat()) {
    throw std::runtime_error("insert() Error, GPU vector cant use operator[]");
  }
  if (get_row() < i) {
    throw std::runtime_error("insert() Error, A.row < i");
  }
  if (get_col() < j) {
    throw std::runtime_error("insert() Error, A.col < j");
  }
  val[get_col() * i + j] = Val;
}
template void Dense<double>::insert(const size_t i, const size_t j,
                                    const double Val);
template void Dense<float>::insert(const size_t i, const size_t j,
                                   const float Val);

} // namespace matrix
} // namespace monolish
