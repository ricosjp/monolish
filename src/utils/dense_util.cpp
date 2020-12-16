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

// matrix utils ///

template <typename T> void Dense<T>::print_all() {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  if (get_device_mem_stat()) {
    throw std::runtime_error("Error, GPU matrix cant use print_all");
  }

  for (size_t i = 0; i < get_row(); i++) {
    for (size_t j = 0; j < get_col(); j++) {
      std::cout << i + 1 << " " << j + 1 << " " << val[i * get_col() + j]
                << std::endl;
    }
  }

  logger.util_out();
}
template void Dense<double>::print_all();
template void Dense<float>::print_all();

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

template <typename T> Dense<T> &Dense<T>::transpose() {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  Dense<T> B(get_col(), get_row());
  for (size_t i = 0; i < get_row(); ++i) {
    for (size_t j = 0; j < get_col(); ++j) {
      B.val[j * get_row() + i] = val[i * get_col() + j];
    }
  }
  std::copy(B.val.data(), B.val.data() + nnz, val.begin());
  set_row(B.get_row());
  set_col(B.get_col());
  logger.util_out();
  return *this;
}
template Dense<double> &Dense<double>::transpose();
template Dense<float> &Dense<float>::transpose();

template <typename T> void Dense<T>::transpose(const Dense<T> &B) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  set_row(B.get_col());
  set_col(B.get_row());
  val.resize(B.get_row() * B.get_col());

  for (size_t i = 0; i < get_row(); ++i) {
    for (size_t j = 0; j < get_col(); ++j) {
      val[j * get_row() + i] = B.val[i * get_col() + j];
    }
  }
  logger.util_out();
}
template void Dense<double>::transpose(const Dense<double> &B);
template void Dense<float>::transpose(const Dense<float> &B);

// matrix convert ///

template <typename T> void Dense<T>::convert(const COO<T> &coo) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  set_row(coo.get_row());
  set_col(coo.get_col());
  set_nnz(get_row() * get_col());
  val.resize(get_row() * get_col());

#pragma omp parallel for
  for (size_t i = 0; i < get_nnz(); i++) {
    val[i] = 0.0;
  }

  for (size_t i = 0; i < coo.get_nnz(); i++) {
    insert(coo.row_index[i], coo.col_index[i], coo.val[i]);
  }
  logger.util_out();
}
template void Dense<double>::convert(const COO<double> &coo);
template void Dense<float>::convert(const COO<float> &coo);

template <typename T> void Dense<T>::convert(const Dense<T> &mat) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  val.resize(mat.get_nnz());

  rowN = mat.get_row();
  colN = mat.get_col();
  nnz = mat.get_nnz();

  // gpu copy and recv
  if (mat.get_device_mem_stat()) {
    send();

#if MONOLISH_USE_GPU
    T *vald = val.data();
    const T *Mvald = mat.val.data();

#pragma omp target teams distribute parallel for
    for (size_t i = 0; i < nnz; i++) {
      vald[i] = Mvald[i];
    }

    nonfree_recv();
#endif
  } else {
    std::copy(mat.val.data(), mat.val.data() + nnz, val.begin());
  }

  logger.util_out();
}
template void Dense<double>::convert(const Dense<double> &mat);
template void Dense<float>::convert(const Dense<float> &mat);

template <typename T> bool Dense<T>::operator==(const Dense<T> &mat) const {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  if (get_device_mem_stat()) {
    throw std::runtime_error("Error, GPU Dense cant use operator==");
  }

  if (get_row() != mat.get_row()) {
    return false;
  }
  if (get_col() != mat.get_col()) {
    return false;
  }

  if (get_device_mem_stat() != mat.get_device_mem_stat()) {
    return false;
  }

  if (get_device_mem_stat() == true) {
    if (!(internal::vequal(get_nnz(), val.data(), mat.val.data(), true))) {
      return false;
    }
  }

  if (!(internal::vequal(get_nnz(), val.data(), mat.val.data(), false))) {
    return false;
  }

  logger.util_out();
  return true;
}
template bool Dense<double>::operator==(const Dense<double> &mat) const;
template bool Dense<float>::operator==(const Dense<float> &mat) const;

template <typename T> bool Dense<T>::operator!=(const Dense<T> &mat) const {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  if (get_device_mem_stat()) {
    throw std::runtime_error("Error, GPU Dense cant use operator!=");
  }

  if (get_row() != mat.get_row()) {
    return true;
  }
  if (get_col() != mat.get_col()) {
    return true;
  }

  if (get_device_mem_stat() != mat.get_device_mem_stat()) {
    return true;
  }

  if (get_device_mem_stat() == true) {
    if (internal::vequal(get_nnz(), val.data(), mat.val.data(), true)) {
      return false;
    }
  }

  if (internal::vequal(get_nnz(), val.data(), mat.val.data(), false)) {
    return false;
  }

  logger.util_out();
  return true;
}
template bool Dense<double>::operator!=(const Dense<double> &mat) const;
template bool Dense<float>::operator!=(const Dense<float> &mat) const;
} // namespace matrix
} // namespace monolish
