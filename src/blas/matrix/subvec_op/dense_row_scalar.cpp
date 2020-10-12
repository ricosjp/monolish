#include "../../../../include/monolish_blas.hpp"
#include "../../../monolish_internal.hpp"

namespace monolish {
namespace matrix {

// add scalar
template <typename T> void Dense<T>::row_add(const size_t r, const T alpha) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  T *vald = val.data();
  const size_t N = get_col();
  const size_t Len = get_col();

#if MONOLISH_USE_GPU // gpu
  size_t nnz = get_nnz();

#pragma omp target teams distribute parallel for
  for (size_t i = 0; i < Len; i++) {
    vald[N * r + i] += alpha;
  }
#else // cpu

#pragma omp parallel for
  for (size_t i = 0; i < Len; i++) {
    vald[N * r + i] += alpha;
  }
#endif

  logger.func_out();
}
template void monolish::matrix::Dense<double>::row_add(const size_t r,
                                                       const double alpha);
template void monolish::matrix::Dense<float>::row_add(const size_t r,
                                                      const float alpha);

// sub scalar
template <typename T> void Dense<T>::row_sub(const size_t r, const T alpha) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  T *vald = val.data();
  const size_t N = get_col();
  const size_t Len = get_col();

#if MONOLISH_USE_GPU // gpu
  size_t nnz = get_nnz();

#pragma omp target teams distribute parallel for
  for (size_t i = 0; i < Len; i++) {
    vald[N * r + i] -= alpha;
  }
#else // cpu

#pragma omp parallel for
  for (size_t i = 0; i < Len; i++) {
    vald[N * r + i] -= alpha;
  }
#endif

  logger.func_out();
}
template void monolish::matrix::Dense<double>::row_sub(const size_t r,
                                                       const double alpha);
template void monolish::matrix::Dense<float>::row_sub(const size_t r,
                                                      const float alpha);

// mul scalar
template <typename T> void Dense<T>::row_mul(const size_t r, const T alpha) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  T *vald = val.data();
  const size_t N = get_col();
  const size_t Len = get_col();

#if MONOLISH_USE_GPU // gpu
  size_t nnz = get_nnz();

#pragma omp target teams distribute parallel for
  for (size_t i = 0; i < Len; i++) {
    vald[N * r + i] *= alpha;
  }
#else // cpu

#pragma omp parallel for
  for (size_t i = 0; i < Len; i++) {
    vald[N * r + i] *= alpha;
  }
#endif

  logger.func_out();
}
template void monolish::matrix::Dense<double>::row_mul(const size_t r,
                                                       const double alpha);
template void monolish::matrix::Dense<float>::row_mul(const size_t r,
                                                      const float alpha);

// div scalar
template <typename T> void Dense<T>::row_div(const size_t r, const T alpha) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  T *vald = val.data();
  const size_t N = get_col();
  const size_t Len = get_col();

#if MONOLISH_USE_GPU // gpu
  size_t nnz = get_nnz();

#pragma omp target teams distribute parallel for
  for (size_t i = 0; i < Len; i++) {
    vald[N * r + i] /= alpha;
  }
#else // cpu

#pragma omp parallel for
  for (size_t i = 0; i < Len; i++) {
    vald[N * r + i] /= alpha;
  }
#endif

  logger.func_out();
}
template void monolish::matrix::Dense<double>::row_div(const size_t r,
                                                       const double alpha);
template void monolish::matrix::Dense<float>::row_div(const size_t r,
                                                      const float alpha);

} // namespace matrix
} // namespace monolish
