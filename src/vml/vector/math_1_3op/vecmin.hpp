#pragma once

namespace monolish {
namespace {
template <typename F1, typename F2, typename F3>
void vvmin_core(const F1 &a, const F2 &b, F3 &y) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(a, b, y));
  assert(util::is_same_device_mem_stat(a, b, y));

  internal::vmin(y.size(), a.begin(), b.begin(), y.begin(),
                 y.get_device_mem_stat());

  logger.func_out();
}

template <typename F1, typename F2, typename F3>
void svmin_core(const F1 &a, const F2 alpha, F3 &y) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(a, y));
  assert(util::is_same_device_mem_stat(a, y));

  internal::vmin(y.size(), a.begin(), alpha, y.begin(),
                 y.get_device_mem_stat());

  logger.func_out();
}

template <typename F1, typename F2> F2 vmin_core(const F1 &y) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  F2 ret = internal::vmin(y.size(), y.begin(), y.get_device_mem_stat());

  logger.func_out();
  return ret;
}
} // namespace
} // namespace monolish
