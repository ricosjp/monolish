#include "../../../../include/monolish_blas.hpp"
#include "../../../monolish_internal.hpp"

namespace monolish {

// double ///////////////////
void blas::matadd(const matrix::CRS<double> &A, const matrix::CRS<double> &B,
                  matrix::CRS<double> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (A.get_row() != B.get_row() && A.get_row() != C.get_row()) {
    throw std::runtime_error("error A.row != B.row != C.row");
  }
  if (A.get_col() != B.get_col() && A.get_col() != C.get_col()) {
    throw std::runtime_error("error A.col != B.col != C.col");
  }
  if (A.get_device_mem_stat() != B.get_device_mem_stat() ||
      A.get_device_mem_stat() != C.get_device_mem_stat()) {
    throw std::runtime_error("error get_device_mem_stat() is not same");
  }

  const double *Ad = A.val.data();
  const double *Bd = B.val.data();
  double *Cd = C.val.data();

  // MN = MK * KN
  const size_t nnz = A.get_nnz();

  if (A.get_device_mem_stat() == true) {
#if MONOLISH_USE_GPU
#pragma omp target teams distribute parallel for
    for (size_t i = 0; i < nnz; i++) {
      Cd[i] = Ad[i] + Bd[i];
    }
#else
    throw std::runtime_error("error USE_GPU is false, but gpu_status == true");
#endif
  } else {
#pragma omp parallel for
    for (size_t i = 0; i < nnz; i++) {
      Cd[i] = Ad[i] + Bd[i];
    }
  }
  logger.func_out();
}

// float ///////////////////
void blas::matadd(const matrix::CRS<float> &A, const matrix::CRS<float> &B,
                  matrix::CRS<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (A.get_row() != B.get_row() && A.get_row() != C.get_row()) {
    throw std::runtime_error("error A.row != B.row != C.row");
  }
  if (A.get_col() != B.get_col() && A.get_col() != C.get_col()) {
    throw std::runtime_error("error A.col != B.col != C.col");
  }
  if (A.get_device_mem_stat() != B.get_device_mem_stat() ||
      A.get_device_mem_stat() != C.get_device_mem_stat()) {
    throw std::runtime_error("error get_device_mem_stat() is not same");
  }

  const float *Ad = A.val.data();
  const float *Bd = B.val.data();
  float *Cd = C.val.data();

  // MN = MK * KN
  const size_t nnz = A.get_nnz();

  if (A.get_device_mem_stat() == true) {
#if MONOLISH_USE_GPU
#pragma omp target teams distribute parallel for
    for (size_t i = 0; i < nnz; i++) {
      Cd[i] = Ad[i] + Bd[i];
    }
#else
    throw std::runtime_error("error USE_GPU is false, but gpu_status == true");
#endif
  } else {
#pragma omp parallel for
    for (size_t i = 0; i < nnz; i++) {
      Cd[i] = Ad[i] + Bd[i];
    }
  }

  logger.func_out();
}

template <typename T>
matrix::CRS<T> matrix::CRS<T>::operator+(const matrix::CRS<T> &B) {
  matrix::CRS<T> C(*this);
  if(gpu_status==true){
      C.send();
  }

  blas::matadd(*this, B, C);

  return C;
}
template matrix::CRS<double>
matrix::CRS<double>::operator+(const matrix::CRS<double> &B);
template matrix::CRS<float>
matrix::CRS<float>::operator+(const matrix::CRS<float> &B);
} // namespace monolish
