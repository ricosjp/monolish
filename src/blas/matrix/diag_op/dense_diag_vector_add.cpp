#include "../../../../include/monolish_blas.hpp"
#include "../../../internal/monolish_internal.hpp"

namespace monolish::matrix {

// add vector

template <typename T> void Dense<T>::diag_add(const vector<T> &vec) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  const T *vecd = vec.data();

  T *vald = val.data();
  const size_t N = get_col();
  const size_t Len = get_row() > get_col() ? get_row() : get_col();

  assert(Len == vec.size());

  if (gpu_status == true) {
#if MONOLISH_USE_NVIDIA_GPU // gpu
#pragma omp target teams distribute parallel for
    for (size_t i = 0; i < Len; i++) {
      vald[N * i + i] += vecd[i];
    }
#else
    throw std::runtime_error("error USE_GPU is false, but gpu_status == true");
#endif
  } else {
#pragma omp parallel for
    for (size_t i = 0; i < Len; i++) {
      vald[N * i + i] += vecd[i];
    }
  }

  logger.func_out();
}
template void
monolish::matrix::Dense<double>::diag_add(const vector<double> &vec);
template void
monolish::matrix::Dense<float>::diag_add(const vector<float> &vec);

template <typename T> void Dense<T>::diag_add(const view1D<vector<T>, T> &vec) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  const T *vecd = vec.data();

  T *vald = val.data();
  const size_t N = get_col();
  const size_t Len = get_row() > get_col() ? get_row() : get_col();

  assert(Len == vec.size());

  if (gpu_status == true) {
#if MONOLISH_USE_NVIDIA_GPU // gpu
#pragma omp target teams distribute parallel for
    for (size_t i = 0; i < Len; i++) {
      vald[N * i + i] += vecd[i];
    }
#else
    throw std::runtime_error("error USE_GPU is false, but gpu_status == true");
#endif
  } else {
#pragma omp parallel for
    for (size_t i = 0; i < Len; i++) {
      vald[N * i + i] += vecd[i];
    }
  }

  logger.func_out();
}
template void monolish::matrix::Dense<double>::diag_add(
    const view1D<vector<double>, double> &vec);
template void monolish::matrix::Dense<float>::diag_add(
    const view1D<vector<float>, float> &vec);

template <typename T>
void Dense<T>::diag_add(const view1D<matrix::Dense<T>, T> &vec) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  const T *vecd = vec.data();

  T *vald = val.data();
  const size_t N = get_col();
  const size_t Len = get_row() > get_col() ? get_row() : get_col();

  assert(Len == vec.size());

  if (gpu_status == true) {
#if MONOLISH_USE_NVIDIA_GPU // gpu
#pragma omp target teams distribute parallel for
    for (size_t i = 0; i < Len; i++) {
      vald[N * i + i] += vecd[i];
    }
#else
    throw std::runtime_error("error USE_GPU is false, but gpu_status == true");
#endif
  } else {
#pragma omp parallel for
    for (size_t i = 0; i < Len; i++) {
      vald[N * i + i] += vecd[i];
    }
  }

  logger.func_out();
}
template void monolish::matrix::Dense<double>::diag_add(
    const view1D<matrix::Dense<double>, double> &vec);
template void monolish::matrix::Dense<float>::diag_add(
    const view1D<matrix::Dense<float>, float> &vec);
} // namespace monolish::matrix
