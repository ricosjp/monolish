#pragma once

namespace monolish {
namespace {
template <typename F1, typename F2, typename F3>
void vvmax_core(const F1 &a, const F2 &b, F3 &y) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(a, b, y));
  assert(util::is_same_device_mem_stat(a, b, y));

  internal::vmax(y.size(), a.begin(), b.begin(), y.begin(),
                 y.get_device_mem_stat());

  logger.func_out();
}

template <typename F1, typename F2, typename F3>
void svmax_core(const F1 &a, const F2 alpha, F3 &y) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(a, y));
  assert(util::is_same_device_mem_stat(a, y));

  internal::vmax(y.size(), a.begin(), alpha, y.begin(),
                 y.get_device_mem_stat());

  logger.func_out();
}

template <typename F1, typename F2> F2 vmax_core(const F1 &y) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  F2 ret = internal::vmax(y.size(), y.begin(), y.get_device_mem_stat());

  logger.func_out();
  return ret;
}
} // namespace
} // namespace monolish
