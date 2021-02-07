#include "../../../include/monolish_vml.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {

// power, sqrt
void vml::pow(const matrix::Dense<float> &A, const matrix::Dense<float> &B,
              matrix::Dense<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(A, B, C));
  assert(util::is_same_device_mem_stat(A, B, C));

  internal::vpow(A.get_nnz(), A.val.data(), B.val.data(), C.val.data(),
                 C.get_device_mem_stat());

  logger.func_out();
}

void vml::pow(const matrix::Dense<float> &A, const float alpha,
              matrix::Dense<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(A, C));
  assert(util::is_same_device_mem_stat(A, C));

  internal::vpow(A.get_nnz(), A.val.data(), alpha, C.val.data(),
                 C.get_device_mem_stat());

  logger.func_out();
}

void vml::sqrt(const matrix::Dense<float> &A, matrix::Dense<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(A, C));
  assert(util::is_same_device_mem_stat(A, C));

  internal::vsqrt(A.get_nnz(), A.val.data(), C.val.data(),
                  A.get_device_mem_stat());

  logger.func_out();
}

void vml::sinh(const matrix::Dense<float> &A, matrix::Dense<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(A, C));
  assert(util::is_same_device_mem_stat(A, C));

  internal::vsinh(A.get_nnz(), A.val.data(), C.val.data(),
                  A.get_device_mem_stat());

  logger.func_out();
}

void vml::asin(const matrix::Dense<float> &A, matrix::Dense<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(A, C));
  assert(util::is_same_device_mem_stat(A, C));

  internal::vasin(A.get_nnz(), A.val.data(), C.val.data(),
                  A.get_device_mem_stat());

  logger.func_out();
}

void vml::asinh(const matrix::Dense<float> &A, matrix::Dense<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(A, C));
  assert(util::is_same_device_mem_stat(A, C));

  internal::vasinh(A.get_nnz(), A.val.data(), C.val.data(),
                   A.get_device_mem_stat());

  logger.func_out();
}

// tan
void vml::tan(const matrix::Dense<float> &A, matrix::Dense<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(A, C));
  assert(util::is_same_device_mem_stat(A, C));

  internal::vtan(A.get_nnz(), A.val.data(), C.val.data(),
                 A.get_device_mem_stat());

  logger.func_out();
}

void vml::tanh(const matrix::Dense<float> &A, matrix::Dense<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(A, C));
  assert(util::is_same_device_mem_stat(A, C));

  internal::vtanh(A.get_nnz(), A.val.data(), C.val.data(),
                  A.get_device_mem_stat());

  logger.func_out();
}

void vml::atan(const matrix::Dense<float> &A, matrix::Dense<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(A, C));
  assert(util::is_same_device_mem_stat(A, C));

  internal::vatan(A.get_nnz(), A.val.data(), C.val.data(),
                  A.get_device_mem_stat());

  logger.func_out();
}

void vml::atanh(const matrix::Dense<float> &A, matrix::Dense<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(A, C));
  assert(util::is_same_device_mem_stat(A, C));

  internal::vatanh(A.get_nnz(), A.val.data(), C.val.data(),
                   A.get_device_mem_stat());

  logger.func_out();
}

// other
void vml::ceil(const matrix::Dense<float> &A, matrix::Dense<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(A, C));
  assert(util::is_same_device_mem_stat(A, C));

  internal::vceil(A.get_nnz(), A.val.data(), C.val.data(),
                  A.get_device_mem_stat());

  logger.func_out();
}

void vml::floor(const matrix::Dense<float> &A, matrix::Dense<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(A, C));
  assert(util::is_same_device_mem_stat(A, C));

  internal::vfloor(A.get_nnz(), A.val.data(), C.val.data(),
                   A.get_device_mem_stat());

  logger.func_out();
}

void vml::sign(const matrix::Dense<float> &A, matrix::Dense<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(A, C));
  assert(util::is_same_device_mem_stat(A, C));

  internal::vsign(A.get_nnz(), A.val.data(), C.val.data(),
                  A.get_device_mem_stat());

  logger.func_out();
}

void vml::reciprocal(const matrix::Dense<float> &A, matrix::Dense<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(A, C));
  assert(util::is_same_device_mem_stat(A, C));

  internal::vreciprocal(A.get_nnz(), A.val.data(), C.val.data(),
                        A.get_device_mem_stat());

  logger.func_out();
}

void vml::max(const matrix::Dense<float> &A, const matrix::Dense<float> &B,
              matrix::Dense<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(A, B, C));
  assert(util::is_same_device_mem_stat(A, B, C));

  internal::vmax(C.get_nnz(), A.val.data(), B.val.data(), C.val.data(),
                 C.get_device_mem_stat());

  logger.func_out();
}

float vml::max(const matrix::Dense<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  float max =
      internal::vmax(C.get_nnz(), C.val.data(), C.get_device_mem_stat());

  logger.func_out();
  return max;
}

void vml::min(const matrix::Dense<float> &A, const matrix::Dense<float> &B,
              matrix::Dense<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(A, B, C));
  assert(util::is_same_device_mem_stat(A, B, C));

  internal::vmin(C.get_nnz(), A.val.data(), B.val.data(), C.val.data(),
                 C.get_device_mem_stat());

  logger.func_out();
}

float vml::min(const matrix::Dense<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  float min =
      internal::vmin(C.get_nnz(), C.val.data(), C.get_device_mem_stat());

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
  assert(util::is_same_size(A, B, C));
  assert(util::is_same_structure(A, B, C));
  assert(util::is_same_device_mem_stat(A, B, C));

  internal::vpow(A.get_nnz(), A.val.data(), B.val.data(), C.val.data(),
                 C.get_device_mem_stat());

  logger.func_out();
}

void vml::pow(const matrix::CRS<float> &A, const float alpha,
              matrix::CRS<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(A, C));
  assert(util::is_same_structure(A, C));
  assert(util::is_same_device_mem_stat(A, C));

  internal::vpow(A.get_nnz(), A.val.data(), alpha, C.val.data(),
                 C.get_device_mem_stat());

  logger.func_out();
}

void vml::sqrt(const matrix::CRS<float> &A, matrix::CRS<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(A, C));
  assert(util::is_same_structure(A, C));
  assert(util::is_same_device_mem_stat(A, C));

  internal::vsqrt(A.get_nnz(), A.val.data(), C.val.data(),
                  A.get_device_mem_stat());

  logger.func_out();
}

void vml::sinh(const matrix::CRS<float> &A, matrix::CRS<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(A, C));
  assert(util::is_same_structure(A, C));
  assert(util::is_same_device_mem_stat(A, C));

  internal::vsinh(A.get_nnz(), A.val.data(), C.val.data(),
                  A.get_device_mem_stat());

  logger.func_out();
}

void vml::asin(const matrix::CRS<float> &A, matrix::CRS<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(A, C));
  assert(util::is_same_structure(A, C));
  assert(util::is_same_device_mem_stat(A, C));

  internal::vasin(A.get_nnz(), A.val.data(), C.val.data(),
                  A.get_device_mem_stat());

  logger.func_out();
}

void vml::asinh(const matrix::CRS<float> &A, matrix::CRS<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(A, C));
  assert(util::is_same_structure(A, C));
  assert(util::is_same_device_mem_stat(A, C));

  internal::vasinh(A.get_nnz(), A.val.data(), C.val.data(),
                   A.get_device_mem_stat());

  logger.func_out();
}

// tan
void vml::tan(const matrix::CRS<float> &A, matrix::CRS<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(A, C));
  assert(util::is_same_structure(A, C));
  assert(util::is_same_device_mem_stat(A, C));

  internal::vtan(A.get_nnz(), A.val.data(), C.val.data(),
                 A.get_device_mem_stat());

  logger.func_out();
}

void vml::tanh(const matrix::CRS<float> &A, matrix::CRS<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(A, C));
  assert(util::is_same_structure(A, C));
  assert(util::is_same_device_mem_stat(A, C));

  internal::vtanh(A.get_nnz(), A.val.data(), C.val.data(),
                  A.get_device_mem_stat());

  logger.func_out();
}

void vml::atan(const matrix::CRS<float> &A, matrix::CRS<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(A, C));
  assert(util::is_same_structure(A, C));
  assert(util::is_same_device_mem_stat(A, C));

  internal::vatan(A.get_nnz(), A.val.data(), C.val.data(),
                  A.get_device_mem_stat());

  logger.func_out();
}

void vml::atanh(const matrix::CRS<float> &A, matrix::CRS<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(A, C));
  assert(util::is_same_structure(A, C));
  assert(util::is_same_device_mem_stat(A, C));

  internal::vatanh(A.get_nnz(), A.val.data(), C.val.data(),
                   A.get_device_mem_stat());

  logger.func_out();
}

// other
void vml::ceil(const matrix::CRS<float> &A, matrix::CRS<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(A, C));
  assert(util::is_same_structure(A, C));
  assert(util::is_same_device_mem_stat(A, C));

  internal::vceil(A.get_nnz(), A.val.data(), C.val.data(),
                  A.get_device_mem_stat());

  logger.func_out();
}

void vml::floor(const matrix::CRS<float> &A, matrix::CRS<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(A, C));
  assert(util::is_same_structure(A, C));
  assert(util::is_same_device_mem_stat(A, C));

  internal::vfloor(A.get_nnz(), A.val.data(), C.val.data(),
                   A.get_device_mem_stat());

  logger.func_out();
}

void vml::sign(const matrix::CRS<float> &A, matrix::CRS<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(A, C));
  assert(util::is_same_structure(A, C));
  assert(util::is_same_device_mem_stat(A, C));

  internal::vsign(A.get_nnz(), A.val.data(), C.val.data(),
                  A.get_device_mem_stat());

  logger.func_out();
}

void vml::reciprocal(const matrix::CRS<float> &A, matrix::CRS<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(A, C));
  assert(util::is_same_structure(A, C));
  assert(util::is_same_device_mem_stat(A, C));

  internal::vreciprocal(A.get_nnz(), A.val.data(), C.val.data(),
                        A.get_device_mem_stat());

  logger.func_out();
}

void vml::max(const matrix::CRS<float> &A, const matrix::CRS<float> &B,
              matrix::CRS<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(A, B, C));
  assert(util::is_same_structure(A, B, C));
  assert(util::is_same_device_mem_stat(A, B, C));

  internal::vmax(C.get_nnz(), A.val.data(), B.val.data(), C.val.data(),
                 C.get_device_mem_stat());

  logger.func_out();
}

float vml::max(const matrix::CRS<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  float max =
      internal::vmax(C.get_nnz(), C.val.data(), C.get_device_mem_stat());

  logger.func_out();
  return std::max(max, (float)0.0);
}

void vml::min(const matrix::CRS<float> &A, const matrix::CRS<float> &B,
              matrix::CRS<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(A, B, C));
  assert(util::is_same_structure(A, B, C));
  assert(util::is_same_device_mem_stat(A, B, C));

  internal::vmin(C.get_nnz(), A.val.data(), B.val.data(), C.val.data(),
                 C.get_device_mem_stat());

  logger.func_out();
}

float vml::min(const matrix::CRS<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  float min =
      internal::vmin(C.get_nnz(), C.val.data(), C.get_device_mem_stat());

  logger.func_out();
  return std::min(min, (float)0.0);
}
} // namespace monolish
