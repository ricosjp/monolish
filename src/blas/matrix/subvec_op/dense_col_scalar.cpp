#include "../../../../include/monolish_blas.hpp"
#include "../../../internal/monolish_internal.hpp"

namespace monolish {
namespace matrix {

// add scalar
template <typename T> void Dense<T>::col_add(const size_t c, const T alpha) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  T *vald = val.data();
  const size_t N = get_col();
  const size_t Len = get_row();

  if (gpu_status == true) {
#if MONOLISH_USE_GPU // gpu
#pragma omp target teams distribute parallel for
    for (size_t i = 0; i < Len; i++) {
      vald[N * i + c] += alpha;
    }
#else
    throw std::runtime_error("error USE_GPU is false, but gpu_status == true");
#endif
  } else {
#pragma omp parallel for
    for (size_t i = 0; i < Len; i++) {
      vald[N * i + c] += alpha;
    }
  }

  logger.func_out();
}
template void monolish::matrix::Dense<double>::col_add(const size_t c,
                                                       const double alpha);
template void monolish::matrix::Dense<float>::col_add(const size_t c,
                                                      const float alpha);

// sub scalar
template <typename T> void Dense<T>::col_sub(const size_t c, const T alpha) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  T *vald = val.data();
  const size_t N = get_col();
  const size_t Len = get_row();

  if (gpu_status == true) {
#if MONOLISH_USE_GPU // gpu
#pragma omp target teams distribute parallel for
    for (size_t i = 0; i < Len; i++) {
      vald[N * i + c] -= alpha;
    }
#else
    throw std::runtime_error("error USE_GPU is false, but gpu_status == true");
#endif
  } else {
#pragma omp parallel for
    for (size_t i = 0; i < Len; i++) {
      vald[N * i + c] -= alpha;
    }
  }

  logger.func_out();
}
template void monolish::matrix::Dense<double>::col_sub(const size_t c,
                                                       const double alpha);
template void monolish::matrix::Dense<float>::col_sub(const size_t c,
                                                      const float alpha);

// mul scalar
template <typename T> void Dense<T>::col_mul(const size_t c, const T alpha) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  T *vald = val.data();
  const size_t N = get_col();
  const size_t Len = get_row();

  if (gpu_status == true) {
#if MONOLISH_USE_GPU // gpu
#pragma omp target teams distribute parallel for
    for (size_t i = 0; i < Len; i++) {
      vald[N * i + c] *= alpha;
    }
#else
    throw std::runtime_error("error USE_GPU is false, but gpu_status == true");
#endif
  } else {
#pragma omp parallel for
    for (size_t i = 0; i < Len; i++) {
      vald[N * i + c] *= alpha;
    }
  }

  logger.func_out();
}
template void monolish::matrix::Dense<double>::col_mul(const size_t c,
                                                       const double alpha);
template void monolish::matrix::Dense<float>::col_mul(const size_t c,
                                                      const float alpha);

// div scalar
template <typename T> void Dense<T>::col_div(const size_t c, const T alpha) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  T *vald = val.data();
  const size_t N = get_col();
  const size_t Len = get_row();

  if (gpu_status == true) {
#if MONOLISH_USE_GPU // gpu
#pragma omp target teams distribute parallel for
    for (size_t i = 0; i < Len; i++) {
      vald[N * i + c] /= alpha;
    }
#else
    throw std::runtime_error("error USE_GPU is false, but gpu_status == true");
#endif
  } else {
#pragma omp parallel for
    for (size_t i = 0; i < Len; i++) {
      vald[N * i + c] /= alpha;
    }
  }

  logger.func_out();
}
template void monolish::matrix::Dense<double>::col_div(const size_t c,
                                                       const double alpha);
template void monolish::matrix::Dense<float>::col_div(const size_t c,
                                                      const float alpha);

} // namespace matrix
} // namespace monolish
