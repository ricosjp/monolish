#include "../../../../include/monolish_blas.hpp"
#include "../../../monolish_internal.hpp"

namespace monolish {
namespace matrix {

// add scalar
template <typename T> void Dense<T>::col_add(const size_t c, const T alpha) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  T *vald = val.data();
  const size_t N = get_col();
  const size_t Len = get_row();

#if USE_GPU // gpu

size_t nnz = get_nnz();

#pragma acc data present(vald [0:nnz])
#pragma acc parallel
  {
#pragma acc loop independent
    for (size_t i = 0; i < Len; i++) {
      vald[N * i + c] += alpha;
    }
  }
#else // cpu

#pragma omp parallel for
  for (size_t i = 0; i < Len; i++) {
    vald[N * i + c] += alpha;
  }
#endif

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

#if USE_GPU // gpu
size_t nnz = get_nnz();

#pragma acc data present(vald [0:nnz])
#pragma acc parallel
  {
#pragma acc loop independent
    for (size_t i = 0; i < Len; i++) {
      vald[N * i + c] -= alpha;
    }
  }
#else // cpu

#pragma omp parallel for
  for (size_t i = 0; i < Len; i++) {
    vald[N * i + c] -= alpha;
  }
#endif

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

#if USE_GPU // gpu
  size_t nnz = get_nnz();

#pragma acc data present(vald [0:nnz])
#pragma acc parallel
  {
#pragma acc loop independent
    for (size_t i = 0; i < Len; i++) {
      vald[N * i + c] *= alpha;
    }
  }
#else // cpu

#pragma omp parallel for
  for (size_t i = 0; i < Len; i++) {
    vald[N * i + c] *= alpha;
  }
#endif

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

#if USE_GPU // gpu
  size_t nnz = get_nnz();

#pragma acc data present(vald [0:nnz])
#pragma acc parallel
  {
#pragma acc loop independent
    for (size_t i = 0; i < Len; i++) {
      vald[N * i + c] /= alpha;
    }
  }
#else // cpu

#pragma omp parallel for
  for (size_t i = 0; i < Len; i++) {
    vald[N * i + c] /= alpha;
  }
#endif

  logger.func_out();
}
template void monolish::matrix::Dense<double>::col_div(const size_t c,
                                                       const double alpha);
template void monolish::matrix::Dense<float>::col_div(const size_t c,
                                                      const float alpha);

} // namespace matrix
} // namespace monolish
