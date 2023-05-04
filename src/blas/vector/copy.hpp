#pragma once

namespace monolish {
namespace {
template <typename F1, typename F2> void copy_core(const F1 &x, F2 &y) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  // err
  assert(util::is_same_size(x, y));
  assert(util::is_same_device_mem_stat(x, y));

  internal::vcopy(y.size(), x.begin(), y.begin(), y.get_device_mem_stat());

  logger.util_out();
}
} // namespace
} // namespace monolish
