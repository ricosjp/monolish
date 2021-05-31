#include "../../../../include/monolish_blas.hpp"
#include "../../../internal/monolish_internal.hpp"

namespace monolish {

// double ///////////////////
void blas::matmul(const matrix::LinearOperator<double> &A,
                  const matrix::Dense<double> &B, matrix::Dense<double> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(A.get_col() == B.get_row());
  assert(A.get_row() == C.get_row());
  assert(B.get_col() == C.get_col());
  assert(util::is_same_device_mem_stat(A, B, C));

  if (A.get_matmul_dense_init_flag()) {
    C = A.get_matmul_dense()(B);
  } else {
    throw std::runtime_error("error matmul is not initialized");
  }

  logger.func_out();
}

// float ///////////////////
void blas::matmul(const matrix::LinearOperator<float> &A,
                  const matrix::Dense<float> &B, matrix::Dense<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(A.get_col() == B.get_row());
  assert(A.get_row() == C.get_row());
  assert(B.get_col() == C.get_col());
  assert(util::is_same_device_mem_stat(A, B, C));

  if (A.get_matmul_dense_init_flag()) {
    C = A.get_matmul_dense()(B);
  } else {
    throw std::runtime_error("error matmul is not initialized");
  }

  logger.func_out();
}

// double ///////////////////
void blas::rmatmul(const matrix::LinearOperator<double> &A,
                   const matrix::Dense<double> &B, matrix::Dense<double> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(A.get_row() == B.get_row());
  assert(A.get_col() == C.get_row());
  assert(B.get_col() == C.get_col());
  assert(util::is_same_device_mem_stat(A, B, C));

  if (A.get_rmatmul_dense_init_flag()) {
    C = A.get_rmatmul_dense()(B);
  } else {
    throw std::runtime_error("error rmatmul is not initialized");
  }

  logger.func_out();
}

// float ///////////////////
void blas::rmatmul(const matrix::LinearOperator<float> &A,
                   const matrix::Dense<float> &B, matrix::Dense<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(A.get_row() == B.get_row());
  assert(A.get_col() == C.get_row());
  assert(B.get_col() == C.get_col());
  assert(util::is_same_device_mem_stat(A, B, C));

  if (A.get_rmatmul_dense_init_flag()) {
    C = A.get_rmatmul_dense()(B);
  } else {
    throw std::runtime_error("error rmatmul is not initialized");
  }

  logger.func_out();
}

} // namespace monolish
