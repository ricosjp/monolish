#pragma once

namespace monolish {
namespace {
template <typename F1, typename F2, typename F3>
void ttmax_core(const F1 &A, const F2 &B, F3 &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(A, B, C));
  assert(util::is_same_structure(A, B, C));
  assert(util::is_same_device_mem_stat(A, B, C));

  internal::vmax(C.get_nnz(), A.data(), B.data(), C.data(),
                 C.get_device_mem_stat());

  logger.func_out();
}

template <typename F1, typename F2, typename F3>
void stmax_core(const F1 &A, const F2 alpha, F3 &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(A, C));
  assert(util::is_same_structure(A, C));
  assert(util::is_same_device_mem_stat(A, C));

  internal::vmax(C.get_nnz(), A.data(), alpha, C.data(),
                 C.get_device_mem_stat());

  logger.func_out();
}

template <typename F1, typename F2> F2 tmax_core(const F1 &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  F2 max = internal::vmax(C.get_nnz(), C.data(), C.get_device_mem_stat());

  logger.func_out();
  if (typeid(F1) == typeid(tensor::tensor_Dense<F2>)) {
    return max;
  } else {
    return std::max(max, (F2)0.0);
  }
}
} // namespace
} // namespace monolish
