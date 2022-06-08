#pragma once

namespace monolish {
namespace {

template <typename F1, typename F2, typename F3>
void times_scalar_core(const F1 &alpha, const F2 &a, F3 &y) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  const auto aoffset = a.get_offset();
  const auto yoffset = y.get_offset();

  // err
  assert(util::is_same_size(a, y));
  assert(util::is_same_device_mem_stat(a, y));

  internal::vmul(y.size(), a.data() + aoffset, alpha,
                 y.data() + yoffset, y.get_device_mem_stat());

  logger.func_out();
}

template <typename F1, typename F2, typename F3>
void times_vector_core(const F1 &a, const F2 &b, F3 &y) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  const auto aoffset = a.get_offset();
  const auto boffset = b.get_offset();
  const auto yoffset = y.get_offset();

  // err
  assert(util::is_same_size(a, y));
  assert(util::is_same_device_mem_stat(a, y));

  internal::vmul(y.size(), a.data() + aoffset, b.data() + boffset,
                 y.data() + yoffset, y.get_device_mem_stat());

  logger.func_out();
}

} // namespace
} // namespace monolish
