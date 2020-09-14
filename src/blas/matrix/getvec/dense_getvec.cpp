#include "../../../../include/monolish_blas.hpp"
#include "../../../monolish_internal.hpp"

namespace monolish {
namespace matrix {

// diag
template <typename T> void Dense<T>::diag(vector<T> &vec) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  size_t n = get_row() < get_col() ? rowN : colN;
  size_t nnz = get_nnz();
  T *vecd = vec.data();

  const T *vald = val.data();
  const size_t M = get_row();
  const size_t N = get_col();
  const size_t Len = get_row() > get_col() ? get_row() : get_col();

  if (Len != vec.size()) {
    throw std::runtime_error("error A.size != diag.size");
  }

#if USE_GPU // gpu

#pragma acc data present(vecd [0:n], vald [0:nnz])
#pragma acc parallel
  {
#pragma acc loop independent
    for (size_t i = 0; i < Len; i++) {
      vecd[i] = vald[N * i + i];
    }
  }
#else // cpu

#pragma omp parallel for
  for (size_t i = 0; i < Len; i++) {
    vecd[i] = vald[N * i + i];
  }
#endif

  logger.func_out();
}
template void monolish::matrix::Dense<double>::diag(vector<double> &vec);
template void monolish::matrix::Dense<float>::diag(vector<float> &vec);

// get_row
template <typename T> void Dense<T>::row(const size_t r, vector<T> &vec) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  size_t n = get_row();
  size_t nnz = get_nnz();
  T *vecd = vec.data();

  const T *vald = val.data();
  const size_t M = get_row();
  const size_t N = get_col();

  if (N != vec.size()) {
    throw std::runtime_error("error A.size != row.size");
  }

#if USE_GPU // gpu

#pragma acc data present(vecd [0:n], vald [0:nnz])
#pragma acc parallel
  {
#pragma acc loop independent
    for (size_t i = 0; i < N; i++) {
      vecd[i] = vald[r * N + i];
    }
  }
#else // cpu

#pragma omp parallel for
  for (size_t i = 0; i < N; i++) {
    vecd[i] = vald[r * N + i];
  }
#endif

  logger.func_out();
}
template void monolish::matrix::Dense<double>::row(const size_t r,
                                                   vector<double> &vec);
template void monolish::matrix::Dense<float>::row(const size_t r,
                                                  vector<float> &vec);

// get_row
template <typename T> void Dense<T>::col(const size_t c, vector<T> &vec) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  size_t n = get_col();
  size_t nnz = get_nnz();
  T *vecd = vec.data();

  const T *vald = val.data();
  const size_t M = get_row();
  const size_t N = get_col();

  if (M != vec.size()) {
    throw std::runtime_error("error A.size != row.size");
  }

#if USE_GPU // gpu

#pragma acc data present(vecd [0:n], vald [0:nnz])
#pragma acc parallel
  {
#pragma acc loop independent
    for (size_t i = 0; i < M; i++) {
      vecd[i] = vald[i * N + c];
    }
  }
#else // cpu

#pragma omp parallel for
  for (size_t i = 0; i < M; i++) {
    vecd[i] = vald[i * N + c];
  }
#endif

  logger.func_out();
}
template void monolish::matrix::Dense<double>::col(const size_t c,
                                                   vector<double> &vec);
template void monolish::matrix::Dense<float>::col(const size_t c,
                                                  vector<float> &vec);
} // namespace matrix
} // namespace monolish
