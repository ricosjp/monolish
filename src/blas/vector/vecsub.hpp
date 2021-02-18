#pragma once

namespace monolish {
namespace {
template <typename F1, typename F2, typename F3>
void vecsub_core(const F1 &a, const F2 &b, F3 &y) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  const size_t aoffset = a.get_offset();
  const size_t boffset = b.get_offset();
  const size_t yoffset = y.get_offset();

  // err
  assert(util::is_same_size(a, b, y));
  assert(util::is_same_device_mem_stat(a, b, y));

  internal::vsub(y.size(), a.data() + aoffset, b.data() + boffset,
                 y.data() + yoffset, y.get_device_mem_stat());

  logger.func_out();
}
} // namespace
} // namespace monolish
