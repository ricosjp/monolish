#include "../../../../include/monolish_blas.hpp"
#include "../../../internal/monolish_internal.hpp"

namespace monolish {

///////////////////////////////
// addition
///////////////////////////////

// Dense ///////////////////
void blas::matadd(const matrix::Dense<float> &A,
                  const matrix::Dense<float> &B, matrix::Dense<float> &C) {
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

  internal::vadd(A.get_nnz(), A.val.data(), B.val.data(), C.val.data(),
                 A.get_device_mem_stat());

  logger.func_out();
}

// CRS ///////////////////
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

  internal::vadd(A.get_nnz(), A.val.data(), B.val.data(), C.val.data(),
                 A.get_device_mem_stat());

  logger.func_out();
}

template <typename T>
matrix::Dense<T> matrix::Dense<T>::operator+(const matrix::Dense<T> &B) {
  matrix::Dense<T> C(get_row(), get_col());
  if (gpu_status == true) {
    C.send();
  }

  blas::matadd(*this, B, C);

  return C;
}
template matrix::Dense<float>
matrix::Dense<float>::operator+(const matrix::Dense<float> &B);


template <typename T>
matrix::CRS<T> matrix::CRS<T>::operator+(const matrix::CRS<T> &B) {
  matrix::CRS<T> C(*this);
  if (gpu_status == true) {
    C.send();
  }

  blas::matadd(*this, B, C);

  return C;
}
template matrix::CRS<float>
matrix::CRS<float>::operator+(const matrix::CRS<float> &B);

///////////////////////////////
// subtract
///////////////////////////////

// Dense ///////////////////
void blas::matsub(const matrix::Dense<float> &A,
                  const matrix::Dense<float> &B, matrix::Dense<float> &C) {
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

  internal::vsub(A.get_nnz(), A.val.data(), B.val.data(), C.val.data(),
                 A.get_device_mem_stat());

  logger.func_out();
}

// CRS ///////////////////
void blas::matsub(const matrix::CRS<float> &A, const matrix::CRS<float> &B,
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

  internal::vsub(A.get_nnz(), A.val.data(), B.val.data(), C.val.data(),
                 A.get_device_mem_stat());

  logger.func_out();
}

} // namespace monolish
