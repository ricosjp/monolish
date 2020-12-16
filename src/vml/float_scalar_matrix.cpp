#include "../../include/monolish_vml.hpp"
#include "../internal/monolish_internal.hpp"

namespace monolish {

///////////////////////////////////////////
// Dense////////////////////////////////////
///////////////////////////////////////////
void vml::add(const matrix::Dense<float> &A, const float alpha,
              matrix::Dense<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (A.get_row() != C.get_row()) {
    throw std::runtime_error("error A.row != B.row != C.row");
  }
  if (A.get_col() != C.get_col()) {
    throw std::runtime_error("error A.col != B.col != C.col");
  }
  if (A.get_device_mem_stat() != C.get_device_mem_stat()) {
    throw std::runtime_error("error vector get_device_mem_stat() is not same");
  }

  internal::vadd(A.get_nnz(), A.val.data(), alpha, C.val.data(),
                 C.get_device_mem_stat());

  logger.func_out();
}

void vml::sub(const matrix::Dense<float> &A, const float alpha,
              matrix::Dense<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (A.get_row() != C.get_row()) {
    throw std::runtime_error("error A.row != B.row != C.row");
  }
  if (A.get_col() != C.get_col()) {
    throw std::runtime_error("error A.col != B.col != C.col");
  }
  if (A.get_device_mem_stat() != C.get_device_mem_stat()) {
    throw std::runtime_error("error vector get_device_mem_stat() is not same");
  }

  internal::vsub(A.get_nnz(), A.val.data(), alpha, C.val.data(),
                 C.get_device_mem_stat());

  logger.func_out();
}

void vml::mul(const matrix::Dense<float> &A, const float alpha,
              matrix::Dense<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (A.get_row() != C.get_row()) {
    throw std::runtime_error("error A.row != B.row != C.row");
  }
  if (A.get_col() != C.get_col()) {
    throw std::runtime_error("error A.col != B.col != C.col");
  }
  if (A.get_device_mem_stat() != C.get_device_mem_stat()) {
    throw std::runtime_error("error vector get_device_mem_stat() is not same");
  }

  internal::vmul(A.get_nnz(), A.val.data(), alpha, C.val.data(),
                 C.get_device_mem_stat());

  logger.func_out();
}

void vml::div(const matrix::Dense<float> &A, const float alpha,
              matrix::Dense<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (A.get_row() != C.get_row()) {
    throw std::runtime_error("error A.row != B.row != C.row");
  }
  if (A.get_col() != C.get_col()) {
    throw std::runtime_error("error A.col != B.col != C.col");
  }
  if (A.get_device_mem_stat() != C.get_device_mem_stat()) {
    throw std::runtime_error("error vector get_device_mem_stat() is not same");
  }

  internal::vdiv(A.get_nnz(), A.val.data(), alpha, C.val.data(),
                 C.get_device_mem_stat());

  logger.func_out();
}

///////////////////////////////////////////
// CRS////////////////////////////////////
///////////////////////////////////////////
void vml::add(const matrix::CRS<float> &A, const float alpha,
              matrix::CRS<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (A.get_row() != C.get_row()) {
    throw std::runtime_error("error A.row != B.row != C.row");
  }
  if (A.get_col() != C.get_col()) {
    throw std::runtime_error("error A.col != B.col != C.col");
  }
  if (A.get_device_mem_stat() != C.get_device_mem_stat()) {
    throw std::runtime_error("error vector get_device_mem_stat() is not same");
  }

  internal::vadd(A.get_nnz(), A.val.data(), alpha, C.val.data(),
                 C.get_device_mem_stat());

  logger.func_out();
}

void vml::sub(const matrix::CRS<float> &A, const float alpha,
              matrix::CRS<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (A.get_row() != C.get_row()) {
    throw std::runtime_error("error A.row != B.row != C.row");
  }
  if (A.get_col() != C.get_col()) {
    throw std::runtime_error("error A.col != B.col != C.col");
  }
  if (A.get_device_mem_stat() != C.get_device_mem_stat()) {
    throw std::runtime_error("error vector get_device_mem_stat() is not same");
  }

  internal::vsub(A.get_nnz(), A.val.data(), alpha, C.val.data(),
                 C.get_device_mem_stat());

  logger.func_out();
}

void vml::mul(const matrix::CRS<float> &A, const float alpha,
              matrix::CRS<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (A.get_row() != C.get_row()) {
    throw std::runtime_error("error A.row != B.row != C.row");
  }
  if (A.get_col() != C.get_col()) {
    throw std::runtime_error("error A.col != B.col != C.col");
  }
  if (A.get_device_mem_stat() != C.get_device_mem_stat()) {
    throw std::runtime_error("error vector get_device_mem_stat() is not same");
  }

  internal::vmul(A.get_nnz(), A.val.data(), alpha, C.val.data(),
                 C.get_device_mem_stat());

  logger.func_out();
}

void vml::div(const matrix::CRS<float> &A, const float alpha,
              matrix::CRS<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (A.get_row() != C.get_row()) {
    throw std::runtime_error("error A.row != B.row != C.row");
  }
  if (A.get_col() != C.get_col()) {
    throw std::runtime_error("error A.col != B.col != C.col");
  }
  if (A.get_device_mem_stat() != C.get_device_mem_stat()) {
    throw std::runtime_error("error vector get_device_mem_stat() is not same");
  }

  internal::vdiv(A.get_nnz(), A.val.data(), alpha, C.val.data(),
                 C.get_device_mem_stat());

  logger.func_out();
}

} // namespace monolish
