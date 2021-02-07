#include "../../../include/monolish_vml.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {

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
void vml::max(const matrix::CRS<float> &A, const matrix::CRS<float> &B,
              matrix::CRS<float> &C) {
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
