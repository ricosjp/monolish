#pragma once

namespace monolish {
namespace {
template <typename F1, typename F2, typename F3>
void mmmax_core(const F1 &A, const F2 &B, F3 &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(A, B, C));
  assert(util::is_same_structure(A, B, C));
  assert(util::is_same_device_mem_stat(A, B, C));

  internal::vmax(C.get_nnz(), A.begin(), B.begin(), C.begin(),
                 C.get_device_mem_stat());

  logger.func_out();
}

template <typename F1, typename F2, typename F3>
void smmax_core(const F1 &A, const F2 alpha, F3 &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(A, C));
  assert(util::is_same_structure(A, C));
  assert(util::is_same_device_mem_stat(A, C));

  internal::vmax(C.get_nnz(), A.begin(), alpha, C.begin(),
                 C.get_device_mem_stat());

  logger.func_out();
}

template <typename F1, typename F2> F2 mmax_core(const F1 &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  F2 max = internal::vmax(C.get_nnz(), C.begin(), C.get_device_mem_stat());

  logger.func_out();
  if (typeid(F1) == typeid(matrix::Dense<F2>)) {
    return max;
  } else {
    return std::max(max, (F2)0.0);
  }
}
} // namespace
} // namespace monolish
