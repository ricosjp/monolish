#pragma once

namespace monolish {
namespace {
template <typename F1, typename F2, typename F3, typename F4>
void malo_core(const F1 &A, F2 alpha, F3 beta, F4 &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(A, C));
  assert(util::is_same_structure(A, C));
  assert(util::is_same_device_mem_stat(A, C));

  internal::valo(A.get_nnz(), A.val.data(), alpha, beta, C.val.data(),
                 A.get_device_mem_stat());

  logger.func_out();
}
} // namespace
} // namespace monolish
