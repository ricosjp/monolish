#include "../../include/monolish_vml.hpp"
#include "../internal/monolish_internal.hpp"

namespace monolish {

///////////////////////////////////////////
// Dense////////////////////////////////////
///////////////////////////////////////////
void vml::add(const matrix::Dense<double> &A, const double alpha,
              matrix::Dense<double> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(A, C));
  assert(util::is_same_device_mem_stat(A, C));

  internal::vadd(A.get_nnz(), A.val.data(), alpha, C.val.data(),
                 C.get_device_mem_stat());

  logger.func_out();
}

void vml::sub(const matrix::Dense<double> &A, const double alpha,
              matrix::Dense<double> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(A, C));
  assert(util::is_same_device_mem_stat(A, C));

  internal::vsub(A.get_nnz(), A.val.data(), alpha, C.val.data(),
                 C.get_device_mem_stat());

  logger.func_out();
}

void vml::mul(const matrix::Dense<double> &A, const double alpha,
              matrix::Dense<double> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(A, C));
  assert(util::is_same_device_mem_stat(A, C));

  internal::vmul(A.get_nnz(), A.val.data(), alpha, C.val.data(),
                 C.get_device_mem_stat());

  logger.func_out();
}

void vml::div(const matrix::Dense<double> &A, const double alpha,
              matrix::Dense<double> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(A, C));
  assert(util::is_same_device_mem_stat(A, C));

  internal::vdiv(A.get_nnz(), A.val.data(), alpha, C.val.data(),
                 C.get_device_mem_stat());

  logger.func_out();
}

///////////////////////////////////////////
// CRS////////////////////////////////////
///////////////////////////////////////////
void vml::add(const matrix::CRS<double> &A, const double alpha,
              matrix::CRS<double> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(A, C));
  assert(util::is_same_structure(A, C));
  assert(util::is_same_device_mem_stat(A, C));

  internal::vadd(A.get_nnz(), A.val.data(), alpha, C.val.data(),
                 C.get_device_mem_stat());

  logger.func_out();
}

void vml::sub(const matrix::CRS<double> &A, const double alpha,
              matrix::CRS<double> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(A, C));
  assert(util::is_same_structure(A, C));
  assert(util::is_same_device_mem_stat(A, C));

  internal::vsub(A.get_nnz(), A.val.data(), alpha, C.val.data(),
                 C.get_device_mem_stat());

  logger.func_out();
}

void vml::mul(const matrix::CRS<double> &A, const double alpha,
              matrix::CRS<double> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(A, C));
  assert(util::is_same_structure(A, C));
  assert(util::is_same_device_mem_stat(A, C));

  internal::vmul(A.get_nnz(), A.val.data(), alpha, C.val.data(),
                 C.get_device_mem_stat());

  logger.func_out();
}

void vml::div(const matrix::CRS<double> &A, const double alpha,
              matrix::CRS<double> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(A, C));
  assert(util::is_same_structure(A, C));
  assert(util::is_same_device_mem_stat(A, C));

  internal::vdiv(A.get_nnz(), A.val.data(), alpha, C.val.data(),
                 C.get_device_mem_stat());

  logger.func_out();
}

} // namespace monolish
