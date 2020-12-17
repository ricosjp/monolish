#include "../../include/monolish_vml.hpp"
#include "../internal/monolish_internal.hpp"

namespace monolish {

///////////////////////////////////////////
// Dense////////////////////////////////////
///////////////////////////////////////////
void vml::add(const matrix::Dense<double> &A, const matrix::Dense<double> &B,
              matrix::Dense<double> &C) {
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
    throw std::runtime_error("error matrix get_device_mem_stat() is not same");
  }

  internal::vadd(A.get_nnz(), A.val.data(), B.val.data(), C.val.data(),
                 C.get_device_mem_stat());

  logger.func_out();
}

void vml::sub(const matrix::Dense<double> &A, const matrix::Dense<double> &B,
              matrix::Dense<double> &C) {
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
    throw std::runtime_error("error matrix get_device_mem_stat() is not same");
  }

  internal::vsub(A.get_nnz(), A.val.data(), B.val.data(), C.val.data(),
                 C.get_device_mem_stat());

  logger.func_out();
}

void vml::mul(const matrix::Dense<double> &A, const matrix::Dense<double> &B,
              matrix::Dense<double> &C) {
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
    throw std::runtime_error("error matrix get_device_mem_stat() is not same");
  }

  internal::vmul(A.get_nnz(), A.val.data(), B.val.data(), C.val.data(),
                 C.get_device_mem_stat());

  logger.func_out();
}

void vml::div(const matrix::Dense<double> &A, const matrix::Dense<double> &B,
              matrix::Dense<double> &C) {
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
    throw std::runtime_error("error matrix get_device_mem_stat() is not same");
  }

  internal::vdiv(A.get_nnz(), A.val.data(), B.val.data(), C.val.data(),
                 C.get_device_mem_stat());

  logger.func_out();
}

///////////////////////////////////////////
// CRS////////////////////////////////////
///////////////////////////////////////////
void vml::add(const matrix::CRS<double> &A, const matrix::CRS<double> &B,
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
    throw std::runtime_error("error matrix get_device_mem_stat() is not same");
  }

  internal::vadd(A.get_nnz(), A.val.data(), B.val.data(), C.val.data(),
                 C.get_device_mem_stat());

  logger.func_out();
}

void vml::sub(const matrix::CRS<double> &A, const matrix::CRS<double> &B,
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
    throw std::runtime_error("error matrix get_device_mem_stat() is not same");
  }

  internal::vsub(A.get_nnz(), A.val.data(), B.val.data(), C.val.data(),
                 C.get_device_mem_stat());

  logger.func_out();
}

void vml::mul(const matrix::CRS<double> &A, const matrix::CRS<double> &B,
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
    throw std::runtime_error("error matrix get_device_mem_stat() is not same");
  }

  internal::vmul(A.get_nnz(), A.val.data(), B.val.data(), C.val.data(),
                 C.get_device_mem_stat());

  logger.func_out();
}

void vml::div(const matrix::CRS<double> &A, const matrix::CRS<double> &B,
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
    throw std::runtime_error("error matrix get_device_mem_stat() is not same");
  }

  internal::vdiv(A.get_nnz(), A.val.data(), B.val.data(), C.val.data(),
                 C.get_device_mem_stat());

  logger.func_out();
}

//////////////////////////////////////////
// math
//////////////////////////////////////////
//
void vml::tanh(const matrix::Dense<double> &A, matrix::Dense<double> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (A.get_row() != C.get_row()) {
    throw std::runtime_error("error A.row != C.row");
  }
  if (A.get_col() != C.get_col()) {
    throw std::runtime_error("error A.col != C.col");
  }
  if (A.get_device_mem_stat() != C.get_device_mem_stat()) {
    throw std::runtime_error("error matrix get_device_mem_stat() is not same");
  }

  internal::vtanh(A.get_nnz(), A.val.data(),  C.val.data(), C.get_device_mem_stat());

  logger.func_out();
}

void vml::tanh(const matrix::CRS<double> &A, matrix::CRS<double> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (A.get_row() != C.get_row()) {
    throw std::runtime_error("error A.row != C.row");
  }
  if (A.get_col() != C.get_col()) {
    throw std::runtime_error("error A.col != C.col");
  }
  if (A.get_device_mem_stat() != C.get_device_mem_stat()) {
    throw std::runtime_error("error matrix get_device_mem_stat() is not same");
  }

  internal::vtanh(A.get_nnz(), A.val.data(),  C.val.data(), C.get_device_mem_stat());

  logger.func_out();
}

} // namespace monolish
