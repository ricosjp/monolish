#include "../../../include/monolish_blas.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {

void tanh_core(size_t N, double *vec) {
#if MONOLISH_USE_MKL
  vdTanh(N, vec, vec);
#else
#pragma omp parallel for
  for (size_t i = 0; i < N; i++) {
    vec[i] = std::tanh(vec[i]);
  }
#endif
}

void tanh_core(size_t N, float *vec) {
#if MONOLISH_USE_MKL
  vsTanh(N, vec, vec);
#else
#pragma omp parallel for
  for (size_t i = 0; i < N; i++) {
    vec[i] = std::tanh(vec[i]);
  }
#endif
}

/////////

template <typename T> void vector<T>::tanh() {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  if (gpu_status == true) {
    recv();
  }

  T *vald = val.data();
  tanh_core(get_nnz(), vald);

  if (gpu_status == true) {
    send();
  }

  logger.func_out();
}

template void vector<double>::tanh();
template void vector<float>::tanh();

/////////

template <typename T> void matrix::Dense<T>::tanh() {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  if (gpu_status == true) {
    recv();
  }

  T *vald = val.data();
  tanh_core(get_nnz(), vald);

  if (gpu_status == true) {
    send();
  }

  logger.func_out();
}

template void matrix::Dense<double>::tanh();
template void matrix::Dense<float>::tanh();

/////////

template <typename T> void matrix::CRS<T>::tanh() {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  if (gpu_status == true) {
    recv();
  }

  T *vald = val.data();
  tanh_core(get_nnz(), vald);

  if (gpu_status == true) {
    send();
  }

  logger.func_out();
}

template void matrix::CRS<double>::tanh();
template void matrix::CRS<float>::tanh();

} // namespace monolish
