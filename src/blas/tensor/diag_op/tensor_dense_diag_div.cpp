#include "../../../../include/monolish_blas.hpp"
#include "../../../internal/monolish_internal.hpp"

namespace monolish::tensor {

// div scalar
template <typename T> void tensor_Dense<T>::diag_div(const T alpha) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  T *vald = data();
  size_t N = 1;
  size_t shift = 0;
  size_t Len = get_nnz();
  for(int i=0; i<shape.size(); ++i){
    shift = shift + N;
    N *= shape[i];
    Len = min(Len, shape[i]);
  }

  if (gpu_status == true) {
#if MONOLISH_USE_NVIDIA_GPU // gpu
#pragma omp target teams distribute parallel for
    for (auto i = decltype(Len){0}; i < Len; i++) {
      vald[shift * i] /= alpha;
    }
#else
    throw std::runtime_error("error USE_GPU is false, but gpu_status == true");
#endif
  } else {
#pragma omp parallel for
    for (auto i = decltype(Len){0}; i < Len; i++) {
      vald[shift * i] /= alpha;
    }
  }

  logger.func_out();
}
template void monolish::matrix::Dense<double>::diag_div(const double alpha);
template void monolish::matrix::Dense<float>::diag_div(const float alpha);

// div vector
template <typename T> void Dense<T>::diag_div(const vector<T> &vec) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  const T *vecd = vec.data();

  T *vald = data();
  size_t N = 1;
  size_t shift = 0;
  size_t Len = get_nnz();
  for(int i=0; i<shape.size(); ++i){
    shift = shift + N;
    N *= shape[i];
    Len = min(Len, shape[i]);
  }

  assert(Len == vec.size());

  if (gpu_status == true) {
#if MONOLISH_USE_NVIDIA_GPU // gpu
#pragma omp target teams distribute parallel for
    for (auto i = decltype(Len){0}; i < Len; i++) {
      vald[shift * i] /= vecd[i];
    }
#else
    throw std::runtime_error("error USE_GPU is false, but gpu_status == true");
#endif
  } else {
#pragma omp parallel for
    for (auto i = decltype(Len){0}; i < Len; i++) {
      vald[shift * i] /= vecd[i];
    }
  }

  logger.func_out();
}
template void
monolish::matrix::Dense<double>::diag_div(const vector<double> &vec);
template void
monolish::matrix::Dense<float>::diag_div(const vector<float> &vec);

// div viwe1D<vector>
template <typename T> void Dense<T>::diag_div(const view1D<vector<T>, T> &vec) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  const T *vecd = vec.data();

  T *vald = data();
  size_t N = 1;
  size_t shift = 0;
  size_t Len = get_nnz();
  for(int i=0; i<shape.size(); ++i){
    shift = shift + N;
    N *= shape[i];
    Len = min(Len, shape[i]);
  }

  assert(Len == vec.size());

  if (gpu_status == true) {
#if MONOLISH_USE_NVIDIA_GPU // gpu
#pragma omp target teams distribute parallel for
    for (auto i = decltype(Len){0}; i < Len; i++) {
      vald[shift * i] /= vecd[i];
    }
#else
    throw std::runtime_error("error USE_GPU is false, but gpu_status == true");
#endif
  } else {
#pragma omp parallel for
    for (auto i = decltype(Len){0}; i < Len; i++) {
      vald[shift * i] /= vecd[i];
    }
  }

  logger.func_out();
}
template void monolish::matrix::Dense<double>::diag_div(
    const view1D<vector<double>, double> &vec);
template void monolish::matrix::Dense<float>::diag_div(
    const view1D<vector<float>, float> &vec);

// div viwe1D<Dense>
template <typename T>
void Dense<T>::diag_div(const view1D<matrix::Dense<T>, T> &vec) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  const T *vecd = vec.data();

  T *vald = data();
  size_t N = 1;
  size_t shift = 0;
  size_t Len = get_nnz();
  for(int i=0; i<shape.size(); ++i){
    shift = shift + N;
    N *= shape[i];
    Len = min(Len, shape[i]);
  }

  assert(Len == vec.size());

  if (gpu_status == true) {
#if MONOLISH_USE_NVIDIA_GPU // gpu
#pragma omp target teams distribute parallel for
    for (auto i = decltype(Len){0}; i < Len; i++) {
      vald[shift * i] /= vecd[i];
    }
#else
    throw std::runtime_error("error USE_GPU is false, but gpu_status == true");
#endif
  } else {
#pragma omp parallel for
    for (auto i = decltype(Len){0}; i < Len; i++) {
      vald[shift * i] /= vecd[i];
    }
  }

  logger.func_out();
}
template void monolish::matrix::Dense<double>::diag_div(
    const view1D<matrix::Dense<double>, double> &vec);
template void monolish::matrix::Dense<float>::diag_div(
    const view1D<matrix::Dense<float>, float> &vec);
} // namespace monolish::matrix
