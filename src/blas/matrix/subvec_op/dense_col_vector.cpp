#include "../../../../include/monolish_blas.hpp"
#include "../../../monolish_internal.hpp"

namespace monolish {
namespace matrix {

// add vector
template <typename T> void Dense<T>::row_add(const size_t c, const vector<T>& vec) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  size_t n = get_row() < get_col() ? rowN : colN;
  size_t nnz = get_nnz();
  const T *vecd = vec.data();

  T *vald = val.data();
  const size_t M = get_row();
  const size_t N = get_col();
  const size_t Len = get_row();

  if (Len != vec.size()) {
    throw std::runtime_error("error A.size != diag.size");
  }

#if USE_GPU // gpu

#pragma acc data present(vald [0:nnz], vecd [0:Len])
#pragma acc parallel
  {
#pragma acc loop independent
    for (size_t i = 0; i < Len; i++) {
      vald[N * i + c] += vecd[i];
    }
  }

#else // cpu

#pragma omp parallel for
  for (size_t i = 0; i < Len; i++) {
      vald[N * i + c] += vecd[i];
  }

#endif

  logger.func_out();
}
template void monolish::matrix::Dense<double>::row_add(const size_t c, const vector<double>& vec);
template void monolish::matrix::Dense<float>::row_add(const size_t c, const vector<float>& vec);

// sub vector
template <typename T> void Dense<T>::row_sub(const size_t c, const vector<T>& vec) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  size_t n = get_row() < get_col() ? rowN : colN;
  size_t nnz = get_nnz();
  const T *vecd = vec.data();

  T *vald = val.data();
  const size_t M = get_row();
  const size_t N = get_col();
  const size_t Len = get_row();

  if (Len != vec.size()) {
    throw std::runtime_error("error A.size != diag.size");
  }

#if USE_GPU // gpu

#pragma acc data present(vald [0:nnz], vecd [0:Len])
#pragma acc parallel
  {
#pragma acc loop independent
    for (size_t i = 0; i < Len; i++) {
      vald[N * i + c] -= vecd[i];
    }
  }

#else // cpu

#pragma omp parallel for
  for (size_t i = 0; i < Len; i++) {
      vald[N * i + c] -= vecd[i];
  }

#endif

  logger.func_out();
}
template void monolish::matrix::Dense<double>::row_sub(const size_t c, const vector<double>& vec);
template void monolish::matrix::Dense<float>::row_sub(const size_t c, const vector<float>& vec);

// mul vector
template <typename T> void Dense<T>::row_mul(const size_t c, const vector<T>& vec) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  size_t n = get_row() < get_col() ? rowN : colN;
  size_t nnz = get_nnz();
  const T *vecd = vec.data();

  T *vald = val.data();
  const size_t M = get_row();
  const size_t N = get_col();
  const size_t Len = get_row();

  if (Len != vec.size()) {
    throw std::runtime_error("error A.size != diag.size");
  }

#if USE_GPU // gpu

#pragma acc data present(vald [0:nnz], vecd [0:Len])
#pragma acc parallel
  {
#pragma acc loop independent
    for (size_t i = 0; i < Len; i++) {
      vald[N * i + c] *= vecd[i];
    }
  }

#else // cpu

#pragma omp parallel for
  for (size_t i = 0; i < Len; i++) {
      vald[N * i + c] *= vecd[i];
  }

#endif

  logger.func_out();
}
template void monolish::matrix::Dense<double>::row_mul(const size_t c, const vector<double>& vec);
template void monolish::matrix::Dense<float>::row_mul(const size_t c, const vector<float>& vec);

// div vector
template <typename T> void Dense<T>::row_div(const size_t c, const vector<T>& vec) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  size_t n = get_row() < get_col() ? rowN : colN;
  size_t nnz = get_nnz();
  const T *vecd = vec.data();

  T *vald = val.data();
  const size_t M = get_row();
  const size_t N = get_col();
  const size_t Len = get_row();

  if (Len != vec.size()) {
    throw std::runtime_error("error A.size != diag.size");
  }

#if USE_GPU // gpu

#pragma acc data present(vald [0:nnz], vecd [0:Len])
#pragma acc parallel
  {
#pragma acc loop independent
    for (size_t i = 0; i < Len; i++) {
      vald[N * i + c] /= vecd[i];
    }
  }

#else // cpu

#pragma omp parallel for
  for (size_t i = 0; i < Len; i++) {
      vald[N * i + c] /= vecd[i];
  }

#endif

  logger.func_out();
}
template void monolish::matrix::Dense<double>::row_div(const size_t c, const vector<double>& vec);
template void monolish::matrix::Dense<float>::row_div(const size_t c, const vector<float>& vec);

} // namespace matrix
} // namespace monolish
