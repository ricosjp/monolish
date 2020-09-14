#include "../../../../include/monolish_blas.hpp"
#include "../../../monolish_internal.hpp"

namespace monolish {
namespace matrix {

// add scalar
template <typename T> void Dense<T>::diag_add(const size_t i, const T alpha) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  size_t n = get_row() < get_col() ? rowN : colN;
  size_t nnz = get_nnz();

  T *vald = val.data();
  const size_t M = get_row();
  const size_t N = get_col();
  const size_t Len = get_row() > get_col() ? get_row() : get_col();

#if USE_GPU // gpu

#pragma acc data present(vald [0:nnz])
#pragma acc parallel
  {
#pragma acc loop independent
    for (size_t i = 0; i < Len; i++) {
      vald[N * i + i] += alpha;
    }
  }
#else // cpu

#pragma omp parallel for
  for (size_t i = 0; i < Len; i++) {
      vald[N * i + i] += alpha;
  }
#endif

  logger.func_out();
}
template void monolish::matrix::Dense<double>::diag_add(const size_t i, const double alpha);
template void monolish::matrix::Dense<float>::diag_add(const size_t i, const float alpha);

// sub scalar
template <typename T> void Dense<T>::diag_sub(size_t i, const T alpha) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  size_t n = get_row() < get_col() ? rowN : colN;
  size_t nnz = get_nnz();

  T *vald = val.data();
  const size_t M = get_row();
  const size_t N = get_col();
  const size_t Len = get_row() > get_col() ? get_row() : get_col();

#if USE_GPU // gpu

#pragma acc data present(vald [0:nnz])
#pragma acc parallel
  {
#pragma acc loop independent
    for (size_t i = 0; i < Len; i++) {
      vald[N * i + i] -= alpha;
    }
  }
#else // cpu

#pragma omp parallel for
  for (size_t i = 0; i < Len; i++) {
      vald[N * i + i] -= alpha;
  }
#endif

  logger.func_out();
}
template void monolish::matrix::Dense<double>::diag_sub(const size_t i, const double alpha);
template void monolish::matrix::Dense<float>::diag_sub(const size_t i, const float alpha);

// mul scalar
template <typename T> void Dense<T>::diag_mul(size_t i, const T alpha) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  size_t n = get_row() < get_col() ? rowN : colN;
  size_t nnz = get_nnz();

  T *vald = val.data();
  const size_t M = get_row();
  const size_t N = get_col();
  const size_t Len = get_row() > get_col() ? get_row() : get_col();

#if USE_GPU // gpu

#pragma acc data present(vald [0:nnz])
#pragma acc parallel
  {
#pragma acc loop independent
    for (size_t i = 0; i < Len; i++) {
      vald[N * i + i] *= alpha;
    }
  }
#else // cpu

#pragma omp parallel for
  for (size_t i = 0; i < Len; i++) {
      vald[N * i + i] *= alpha;
  }
#endif

  logger.func_out();
}
template void monolish::matrix::Dense<double>::diag_mul(const size_t i, const double alpha);
template void monolish::matrix::Dense<float>::diag_mul(const size_t i, const float alpha);

// div scalar
template <typename T> void Dense<T>::diag_div(size_t i, const T alpha) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  size_t n = get_row() < get_col() ? rowN : colN;
  size_t nnz = get_nnz();

  T *vald = val.data();
  const size_t M = get_row();
  const size_t N = get_col();
  const size_t Len = get_row() > get_col() ? get_row() : get_col();

#if USE_GPU // gpu

#pragma acc data present(vald [0:nnz])
#pragma acc parallel
  {
#pragma acc loop independent
    for (size_t i = 0; i < Len; i++) {
      vald[N * i + i] /= alpha;
    }
  }
#else // cpu

#pragma omp parallel for
  for (size_t i = 0; i < Len; i++) {
      vald[N * i + i] /= alpha;
  }
#endif

  logger.func_out();
}
template void monolish::matrix::Dense<double>::diag_div(const size_t i, const double alpha);
template void monolish::matrix::Dense<float>::diag_div(const size_t i, const float alpha);

} // namespace matrix
} // namespace monolish
