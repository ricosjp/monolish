#include "../../../include/monolish_vml.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {
namespace {
template <typename F1, typename F2, typename F3>
void mmmin_core(const F1 &A, const F2 &B, F3 &C) {
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

template <typename F1, typename F2> F2 mmin_core(const F1 &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  F2 min = internal::vmin(C.get_nnz(), C.val.data(), C.get_device_mem_stat());

  logger.func_out();
  if (typeid(F1) == typeid(matrix::Dense<F2>)) {
    return min;
  } else {
    return std::min(min, (F2)0.0);
  }
}
} // namespace

namespace vml {
void min(const matrix::Dense<double> &A, const matrix::Dense<double> &B,
         matrix::Dense<double> &C) {
  mmmin_core(A, B, C);
}

void min(const matrix::Dense<float> &A, const matrix::Dense<float> &B,
         matrix::Dense<float> &C) {
  mmmin_core(A, B, C);
}

double min(const matrix::Dense<double> &C) {
  return mmin_core<matrix::Dense<double>, double>(C);
}

float min(const matrix::Dense<float> &C) {
  return mmin_core<matrix::Dense<float>, float>(C);
}

void min(const matrix::CRS<double> &A, const matrix::CRS<double> &B,
         matrix::CRS<double> &C) {
  mmmin_core(A, B, C);
}

void min(const matrix::CRS<float> &A, const matrix::CRS<float> &B,
         matrix::CRS<float> &C) {
  mmmin_core(A, B, C);
}

double min(const matrix::CRS<double> &C) {
  return mmin_core<matrix::CRS<double>, double>(C);
}

float min(const matrix::CRS<float> &C) {
  return mmin_core<matrix::CRS<float>, float>(C);
}
} // namespace vml
} // namespace monolish
