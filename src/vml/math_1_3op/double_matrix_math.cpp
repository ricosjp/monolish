#include "../../../include/monolish_vml.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {

// power, sqrt
void vml::pow(const matrix::Dense<double> &A, const matrix::Dense<double> &B,
              matrix::Dense<double> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(A, B, C));
  assert(util::is_same_device_mem_stat(A, B, C));

  internal::vpow(A.get_nnz(), A.val.data(), B.val.data(), C.val.data(),
                 C.get_device_mem_stat());

  logger.func_out();
}

void vml::pow(const matrix::Dense<double> &A, const double alpha,
              matrix::Dense<double> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(A, C));
  assert(util::is_same_device_mem_stat(A, C));

  internal::vpow(A.get_nnz(), A.val.data(), alpha, C.val.data(),
                 C.get_device_mem_stat());

  logger.func_out();
}

void vml::max(const matrix::Dense<double> &A, const matrix::Dense<double> &B,
              matrix::Dense<double> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(A, B, C));
  assert(util::is_same_device_mem_stat(A, B, C));

  internal::vmax(C.get_nnz(), A.val.data(), B.val.data(), C.val.data(),
                 C.get_device_mem_stat());

  logger.func_out();
}

double vml::max(const matrix::Dense<double> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  double max =
      internal::vmax(C.get_nnz(), C.val.data(), C.get_device_mem_stat());

  logger.func_out();
  return max;
}

void vml::min(const matrix::Dense<double> &A, const matrix::Dense<double> &B,
              matrix::Dense<double> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(A, B, C));
  assert(util::is_same_device_mem_stat(A, B, C));

  internal::vmin(C.get_nnz(), A.val.data(), B.val.data(), C.val.data(),
                 C.get_device_mem_stat());

  logger.func_out();
}

double vml::min(const matrix::Dense<double> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  double min =
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
void vml::pow(const matrix::CRS<double> &A, const matrix::CRS<double> &B,
              matrix::CRS<double> &C) {
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

void vml::pow(const matrix::CRS<double> &A, const double alpha,
              matrix::CRS<double> &C) {
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

void vml::max(const matrix::CRS<double> &A, const matrix::CRS<double> &B,
              matrix::CRS<double> &C) {
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

double vml::max(const matrix::CRS<double> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  double max =
      internal::vmax(C.get_nnz(), C.val.data(), C.get_device_mem_stat());

  logger.func_out();
  return std::max(max, (double)0.0);
}

void vml::min(const matrix::CRS<double> &A, const matrix::CRS<double> &B,
              matrix::CRS<double> &C) {
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

double vml::min(const matrix::CRS<double> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  double min =
      internal::vmin(C.get_nnz(), C.val.data(), C.get_device_mem_stat());

  logger.func_out();
  return std::min(min, (double)0.0);
}
} // namespace monolish
