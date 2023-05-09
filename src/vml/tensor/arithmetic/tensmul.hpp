#pragma once

namespace monolish {

namespace {
template <typename F1, typename F2, typename F3>
void stmul_core(const F1 &A, const F2 alpha, F3 &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(A, C));
  assert(util::is_same_device_mem_stat(A, C));

  internal::vmul(A.get_nnz(), A.begin(), alpha, C.begin(),
                 C.get_device_mem_stat());

  logger.func_out();
}

template <typename F1, typename F2, typename F3>
void ttmul_core(const F1 &A, const F2 &B, F3 &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(A, B, C));
  assert(util::is_same_structure(A, B, C));
  assert(util::is_same_device_mem_stat(A, B, C));

  internal::vmul(A.get_nnz(), A.begin(), B.begin(), C.begin(),
                 C.get_device_mem_stat());

  logger.func_out();
}
} // namespace

} // namespace monolish
