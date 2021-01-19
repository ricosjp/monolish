#include "../../include/monolish_vml.hpp"
#include "../internal/monolish_internal.hpp"

namespace monolish {

// power, sqrt
void vml::pow(const matrix::Dense<float> &A, const matrix::Dense<float> &B,
              matrix::Dense<float> &C) {
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

  internal::vpow(A.get_nnz(), A.val.data(), B.val.data(), C.val.data(),
                 C.get_device_mem_stat());

  logger.func_out();
}

void vml::pow(const matrix::Dense<float> &A, const float alpha,
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

  internal::vpow(A.get_nnz(), A.val.data(), alpha, C.val.data(),
                 C.get_device_mem_stat());

  logger.func_out();
}

void vml::sqrt(const matrix::Dense<float> &A, matrix::Dense<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (A.get_row() != C.get_row()) {
    throw std::runtime_error("error A.row != C.row");
  }
  if (A.get_col() != C.get_col()) {
    throw std::runtime_error("error A.row != C.row");
  }
  if (A.get_device_mem_stat() != C.get_device_mem_stat()) {
    throw std::runtime_error("error matrix get_device_mem_stat() is not same");
  }

  internal::vsqrt(A.get_nnz(), A.val.data(), C.val.data(),
                  A.get_device_mem_stat());

  logger.func_out();
}

// sin
void vml::sin(const matrix::Dense<float> &A, matrix::Dense<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (A.get_row() != C.get_row()) {
    throw std::runtime_error("error A.row != C.row");
  }
  if (A.get_col() != C.get_col()) {
    throw std::runtime_error("error A.row != C.row");
  }
  if (A.get_device_mem_stat() != C.get_device_mem_stat()) {
    throw std::runtime_error("error matrix get_device_mem_stat() is not same");
  }

  internal::vsin(A.get_nnz(), A.val.data(), C.val.data(),
                 A.get_device_mem_stat());

  logger.func_out();
}

void vml::sinh(const matrix::Dense<float> &A, matrix::Dense<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (A.get_row() != C.get_row()) {
    throw std::runtime_error("error A.row != C.row");
  }
  if (A.get_col() != C.get_col()) {
    throw std::runtime_error("error A.row != C.row");
  }
  if (A.get_device_mem_stat() != C.get_device_mem_stat()) {
    throw std::runtime_error("error matrix get_device_mem_stat() is not same");
  }

  internal::vsinh(A.get_nnz(), A.val.data(), C.val.data(),
                  A.get_device_mem_stat());

  logger.func_out();
}

void vml::asin(const matrix::Dense<float> &A, matrix::Dense<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (A.get_row() != C.get_row()) {
    throw std::runtime_error("error A.row != C.row");
  }
  if (A.get_col() != C.get_col()) {
    throw std::runtime_error("error A.row != C.row");
  }
  if (A.get_device_mem_stat() != C.get_device_mem_stat()) {
    throw std::runtime_error("error matrix get_device_mem_stat() is not same");
  }

  internal::vasin(A.get_nnz(), A.val.data(), C.val.data(),
                  A.get_device_mem_stat());

  logger.func_out();
}

void vml::asinh(const matrix::Dense<float> &A, matrix::Dense<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (A.get_row() != C.get_row()) {
    throw std::runtime_error("error A.row != C.row");
  }
  if (A.get_col() != C.get_col()) {
    throw std::runtime_error("error A.row != C.row");
  }
  if (A.get_device_mem_stat() != C.get_device_mem_stat()) {
    throw std::runtime_error("error matrix get_device_mem_stat() is not same");
  }

  internal::vasinh(A.get_nnz(), A.val.data(), C.val.data(),
                   A.get_device_mem_stat());

  logger.func_out();
}

// tan
void vml::tan(const matrix::Dense<float> &A, matrix::Dense<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (A.get_row() != C.get_row()) {
    throw std::runtime_error("error A.row != C.row");
  }
  if (A.get_col() != C.get_col()) {
    throw std::runtime_error("error A.row != C.row");
  }
  if (A.get_device_mem_stat() != C.get_device_mem_stat()) {
    throw std::runtime_error("error matrix get_device_mem_stat() is not same");
  }

  internal::vtan(A.get_nnz(), A.val.data(), C.val.data(),
                 A.get_device_mem_stat());

  logger.func_out();
}

void vml::tanh(const matrix::Dense<float> &A, matrix::Dense<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (A.get_row() != C.get_row()) {
    throw std::runtime_error("error A.row != C.row");
  }
  if (A.get_col() != C.get_col()) {
    throw std::runtime_error("error A.row != C.row");
  }
  if (A.get_device_mem_stat() != C.get_device_mem_stat()) {
    throw std::runtime_error("error matrix get_device_mem_stat() is not same");
  }

  internal::vtanh(A.get_nnz(), A.val.data(), C.val.data(),
                  A.get_device_mem_stat());

  logger.func_out();
}

void vml::atan(const matrix::Dense<float> &A, matrix::Dense<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (A.get_row() != C.get_row()) {
    throw std::runtime_error("error A.row != C.row");
  }
  if (A.get_col() != C.get_col()) {
    throw std::runtime_error("error A.row != C.row");
  }
  if (A.get_device_mem_stat() != C.get_device_mem_stat()) {
    throw std::runtime_error("error matrix get_device_mem_stat() is not same");
  }

  internal::vatan(A.get_nnz(), A.val.data(), C.val.data(),
                  A.get_device_mem_stat());

  logger.func_out();
}

void vml::atanh(const matrix::Dense<float> &A, matrix::Dense<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (A.get_row() != C.get_row()) {
    throw std::runtime_error("error A.row != C.row");
  }
  if (A.get_col() != C.get_col()) {
    throw std::runtime_error("error A.row != C.row");
  }
  if (A.get_device_mem_stat() != C.get_device_mem_stat()) {
    throw std::runtime_error("error matrix get_device_mem_stat() is not same");
  }

  internal::vatanh(A.get_nnz(), A.val.data(), C.val.data(),
                   A.get_device_mem_stat());

  logger.func_out();
}

// other
void vml::ceil(const matrix::Dense<float> &A, matrix::Dense<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (A.get_row() != C.get_row()) {
    throw std::runtime_error("error A.row != C.row");
  }
  if (A.get_col() != C.get_col()) {
    throw std::runtime_error("error A.row != C.row");
  }
  if (A.get_device_mem_stat() != C.get_device_mem_stat()) {
    throw std::runtime_error("error matrix get_device_mem_stat() is not same");
  }

  internal::vceil(A.get_nnz(), A.val.data(), C.val.data(),
                  A.get_device_mem_stat());

  logger.func_out();
}

void vml::floor(const matrix::Dense<float> &A, matrix::Dense<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (A.get_row() != C.get_row()) {
    throw std::runtime_error("error A.row != C.row");
  }
  if (A.get_col() != C.get_col()) {
    throw std::runtime_error("error A.row != C.row");
  }
  if (A.get_device_mem_stat() != C.get_device_mem_stat()) {
    throw std::runtime_error("error matrix get_device_mem_stat() is not same");
  }

  internal::vfloor(A.get_nnz(), A.val.data(), C.val.data(),
                   A.get_device_mem_stat());

  logger.func_out();
}

void vml::sign(const matrix::Dense<float> &A, matrix::Dense<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (A.get_row() != C.get_row()) {
    throw std::runtime_error("error A.row != C.row");
  }
  if (A.get_col() != C.get_col()) {
    throw std::runtime_error("error A.row != C.row");
  }
  if (A.get_device_mem_stat() != C.get_device_mem_stat()) {
    throw std::runtime_error("error matrix get_device_mem_stat() is not same");
  }

  internal::vsign(A.get_nnz(), A.val.data(), C.val.data(),
                  A.get_device_mem_stat());

  logger.func_out();
}

void vml::reciprocal(const matrix::Dense<float> &A, matrix::Dense<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (A.get_row() != C.get_row()) {
    throw std::runtime_error("error A.row != C.row");
  }
  if (A.get_col() != C.get_col()) {
    throw std::runtime_error("error A.row != C.row");
  }
  if (A.get_device_mem_stat() != C.get_device_mem_stat()) {
    throw std::runtime_error("error matrix get_device_mem_stat() is not same");
  }

  internal::vreciprocal(A.get_nnz(), A.val.data(), C.val.data(),
                        A.get_device_mem_stat());

  logger.func_out();
}

void vml::max(const matrix::Dense<float> &A, const matrix::Dense<float> &B, matrix::Dense<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (A.get_row() != C.get_row() || A.get_row() != B.get_row()) {
    throw std::runtime_error("error A.row != C.row");
  }
  if (A.get_col() != C.get_col() || A.get_col() != B.get_col()) {
    throw std::runtime_error("error A.row != C.row");
  }
  if (A.get_device_mem_stat() != C.get_device_mem_stat() || B.get_device_mem_stat() != C.get_device_mem_stat()) {
    throw std::runtime_error("error vector get_device_mem_stat() is not same");
  }

  internal::vmax(C.get_nnz(), A.val.data(), B.val.data(), C.val.data(), C.get_device_mem_stat());

  logger.func_out();
}

float vml::max(const matrix::Dense<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  float max = internal::vmax(C.get_nnz(), C.val.data(), C.get_device_mem_stat());

  logger.func_out();
  return max;
}

void vml::min(const matrix::Dense<float> &A, const matrix::Dense<float> &B, matrix::Dense<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (A.get_row() != C.get_row() || A.get_row() != B.get_row()) {
    throw std::runtime_error("error A.row != C.row");
  }
  if (A.get_col() != C.get_col() || A.get_col() != B.get_col()) {
    throw std::runtime_error("error A.row != C.row");
  }
  if (A.get_device_mem_stat() != C.get_device_mem_stat() || B.get_device_mem_stat() != C.get_device_mem_stat()) {
    throw std::runtime_error("error vector get_device_mem_stat() is not same");
  }

  internal::vmin(C.get_nnz(), A.val.data(), B.val.data(), C.val.data(), C.get_device_mem_stat());

  logger.func_out();
}

float vml::min(const matrix::Dense<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  float min = internal::vmin(C.get_nnz(), C.val.data(), C.get_device_mem_stat());

  logger.func_out();
  return min;
}

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
// CRS
/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

// power, sqrt
void vml::pow(const matrix::CRS<float> &A, const matrix::CRS<float> &B,
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
    throw std::runtime_error("error matrix get_device_mem_stat() is not same");
  }

  internal::vpow(A.get_nnz(), A.val.data(), B.val.data(), C.val.data(),
                 C.get_device_mem_stat());

  logger.func_out();
}

void vml::pow(const matrix::CRS<float> &A, const float alpha,
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

  internal::vpow(A.get_nnz(), A.val.data(), alpha, C.val.data(),
                 C.get_device_mem_stat());

  logger.func_out();
}

void vml::sqrt(const matrix::CRS<float> &A, matrix::CRS<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (A.get_row() != C.get_row()) {
    throw std::runtime_error("error A.row != C.row");
  }
  if (A.get_col() != C.get_col()) {
    throw std::runtime_error("error A.row != C.row");
  }
  if (A.get_device_mem_stat() != C.get_device_mem_stat()) {
    throw std::runtime_error("error matrix get_device_mem_stat() is not same");
  }

  internal::vsqrt(A.get_nnz(), A.val.data(), C.val.data(),
                  A.get_device_mem_stat());

  logger.func_out();
}

// sin
void vml::sin(const matrix::CRS<float> &A, matrix::CRS<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (A.get_row() != C.get_row()) {
    throw std::runtime_error("error A.row != C.row");
  }
  if (A.get_col() != C.get_col()) {
    throw std::runtime_error("error A.row != C.row");
  }
  if (A.get_device_mem_stat() != C.get_device_mem_stat()) {
    throw std::runtime_error("error matrix get_device_mem_stat() is not same");
  }

  internal::vsin(A.get_nnz(), A.val.data(), C.val.data(),
                 A.get_device_mem_stat());

  logger.func_out();
}

void vml::sinh(const matrix::CRS<float> &A, matrix::CRS<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (A.get_row() != C.get_row()) {
    throw std::runtime_error("error A.row != C.row");
  }
  if (A.get_col() != C.get_col()) {
    throw std::runtime_error("error A.row != C.row");
  }
  if (A.get_device_mem_stat() != C.get_device_mem_stat()) {
    throw std::runtime_error("error matrix get_device_mem_stat() is not same");
  }

  internal::vsinh(A.get_nnz(), A.val.data(), C.val.data(),
                  A.get_device_mem_stat());

  logger.func_out();
}

void vml::asin(const matrix::CRS<float> &A, matrix::CRS<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (A.get_row() != C.get_row()) {
    throw std::runtime_error("error A.row != C.row");
  }
  if (A.get_col() != C.get_col()) {
    throw std::runtime_error("error A.row != C.row");
  }
  if (A.get_device_mem_stat() != C.get_device_mem_stat()) {
    throw std::runtime_error("error matrix get_device_mem_stat() is not same");
  }

  internal::vasin(A.get_nnz(), A.val.data(), C.val.data(),
                  A.get_device_mem_stat());

  logger.func_out();
}

void vml::asinh(const matrix::CRS<float> &A, matrix::CRS<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (A.get_row() != C.get_row()) {
    throw std::runtime_error("error A.row != C.row");
  }
  if (A.get_col() != C.get_col()) {
    throw std::runtime_error("error A.row != C.row");
  }
  if (A.get_device_mem_stat() != C.get_device_mem_stat()) {
    throw std::runtime_error("error matrix get_device_mem_stat() is not same");
  }

  internal::vasinh(A.get_nnz(), A.val.data(), C.val.data(),
                   A.get_device_mem_stat());

  logger.func_out();
}

// tan
void vml::tan(const matrix::CRS<float> &A, matrix::CRS<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (A.get_row() != C.get_row()) {
    throw std::runtime_error("error A.row != C.row");
  }
  if (A.get_col() != C.get_col()) {
    throw std::runtime_error("error A.row != C.row");
  }
  if (A.get_device_mem_stat() != C.get_device_mem_stat()) {
    throw std::runtime_error("error matrix get_device_mem_stat() is not same");
  }

  internal::vtan(A.get_nnz(), A.val.data(), C.val.data(),
                 A.get_device_mem_stat());

  logger.func_out();
}

void vml::tanh(const matrix::CRS<float> &A, matrix::CRS<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (A.get_row() != C.get_row()) {
    throw std::runtime_error("error A.row != C.row");
  }
  if (A.get_col() != C.get_col()) {
    throw std::runtime_error("error A.row != C.row");
  }
  if (A.get_device_mem_stat() != C.get_device_mem_stat()) {
    throw std::runtime_error("error matrix get_device_mem_stat() is not same");
  }

  internal::vtanh(A.get_nnz(), A.val.data(), C.val.data(),
                  A.get_device_mem_stat());

  logger.func_out();
}

void vml::atan(const matrix::CRS<float> &A, matrix::CRS<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (A.get_row() != C.get_row()) {
    throw std::runtime_error("error A.row != C.row");
  }
  if (A.get_col() != C.get_col()) {
    throw std::runtime_error("error A.row != C.row");
  }
  if (A.get_device_mem_stat() != C.get_device_mem_stat()) {
    throw std::runtime_error("error matrix get_device_mem_stat() is not same");
  }

  internal::vatan(A.get_nnz(), A.val.data(), C.val.data(),
                  A.get_device_mem_stat());

  logger.func_out();
}

void vml::atanh(const matrix::CRS<float> &A, matrix::CRS<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (A.get_row() != C.get_row()) {
    throw std::runtime_error("error A.row != C.row");
  }
  if (A.get_col() != C.get_col()) {
    throw std::runtime_error("error A.row != C.row");
  }
  if (A.get_device_mem_stat() != C.get_device_mem_stat()) {
    throw std::runtime_error("error matrix get_device_mem_stat() is not same");
  }

  internal::vatanh(A.get_nnz(), A.val.data(), C.val.data(),
                   A.get_device_mem_stat());

  logger.func_out();
}

// other
void vml::ceil(const matrix::CRS<float> &A, matrix::CRS<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (A.get_row() != C.get_row()) {
    throw std::runtime_error("error A.row != C.row");
  }
  if (A.get_col() != C.get_col()) {
    throw std::runtime_error("error A.row != C.row");
  }
  if (A.get_device_mem_stat() != C.get_device_mem_stat()) {
    throw std::runtime_error("error matrix get_device_mem_stat() is not same");
  }

  internal::vceil(A.get_nnz(), A.val.data(), C.val.data(),
                  A.get_device_mem_stat());

  logger.func_out();
}

void vml::floor(const matrix::CRS<float> &A, matrix::CRS<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (A.get_row() != C.get_row()) {
    throw std::runtime_error("error A.row != C.row");
  }
  if (A.get_col() != C.get_col()) {
    throw std::runtime_error("error A.row != C.row");
  }
  if (A.get_device_mem_stat() != C.get_device_mem_stat()) {
    throw std::runtime_error("error matrix get_device_mem_stat() is not same");
  }

  internal::vfloor(A.get_nnz(), A.val.data(), C.val.data(),
                   A.get_device_mem_stat());

  logger.func_out();
}

void vml::sign(const matrix::CRS<float> &A, matrix::CRS<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (A.get_row() != C.get_row()) {
    throw std::runtime_error("error A.row != C.row");
  }
  if (A.get_col() != C.get_col()) {
    throw std::runtime_error("error A.row != C.row");
  }
  if (A.get_device_mem_stat() != C.get_device_mem_stat()) {
    throw std::runtime_error("error matrix get_device_mem_stat() is not same");
  }

  internal::vsign(A.get_nnz(), A.val.data(), C.val.data(),
                  A.get_device_mem_stat());

  logger.func_out();
}

void vml::reciprocal(const matrix::CRS<float> &A, matrix::CRS<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (A.get_row() != C.get_row()) {
    throw std::runtime_error("error A.row != C.row");
  }
  if (A.get_col() != C.get_col()) {
    throw std::runtime_error("error A.row != C.row");
  }
  if (A.get_device_mem_stat() != C.get_device_mem_stat()) {
    throw std::runtime_error("error matrix get_device_mem_stat() is not same");
  }

  internal::vreciprocal(A.get_nnz(), A.val.data(), C.val.data(),
                        A.get_device_mem_stat());

  logger.func_out();
}

void vml::max(const matrix::CRS<float> &A, const matrix::CRS<float> &B, matrix::CRS<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (A.get_row() != C.get_row() || A.get_row() != B.get_row()) {
    throw std::runtime_error("error A.row != C.row");
  }
  if (A.get_col() != C.get_col() || A.get_col() != B.get_col()) {
    throw std::runtime_error("error A.row != C.row");
  }
  if (A.get_device_mem_stat() != C.get_device_mem_stat() || B.get_device_mem_stat() != C.get_device_mem_stat()) {
    throw std::runtime_error("error vector get_device_mem_stat() is not same");
  }

  internal::vmax(C.get_nnz(), A.val.data(), B.val.data(), C.val.data(), C.get_device_mem_stat());

  logger.func_out();
}

float vml::max(const matrix::CRS<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  float max = internal::vmax(C.get_nnz(), C.val.data(), C.get_device_mem_stat());

  logger.func_out();
  return std::max(max,0);
}

void vml::min(const matrix::CRS<float> &A, const matrix::CRS<float> &B, matrix::CRS<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  if (A.get_row() != C.get_row() || A.get_row() != B.get_row()) {
    throw std::runtime_error("error A.row != C.row");
  }
  if (A.get_col() != C.get_col() || A.get_col() != B.get_col()) {
    throw std::runtime_error("error A.row != C.row");
  }
  if (A.get_device_mem_stat() != C.get_device_mem_stat() || B.get_device_mem_stat() != C.get_device_mem_stat()) {
    throw std::runtime_error("error vector get_device_mem_stat() is not same");
  }

  internal::vmin(C.get_nnz(), A.val.data(), B.val.data(), C.val.data(), C.get_device_mem_stat());

  logger.func_out();
}

float vml::min(const matrix::CRS<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  float min = internal::vmin(C.get_nnz(), C.val.data(), C.get_device_mem_stat());

  logger.func_out();
  return std::min(min,0);
}
} // namespace monolish
