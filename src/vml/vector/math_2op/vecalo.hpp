#pragma once

namespace monolish {
namespace {
template <typename F1, typename F2, typename F3, typename F4>
void vecalo_core(const F1 &a, const F2 alpha, const F3 beta, F4 &y) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(a, y));
  assert(util::is_same_device_mem_stat(a, y));

  internal::valo(a.size(), a.begin(), alpha, beta, y.begin(),
                 y.get_device_mem_stat());

  logger.func_out();
}
} // namespace
} // namespace monolish
