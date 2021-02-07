#include "../../../include/monolish_vml.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {
namespace {
template <typename F1, typename F2> void vfloor_core(const F1 &a, F2 &y) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(a, y));
  assert(util::is_same_device_mem_stat(a, y));

  internal::vfloor(y.size(), a.data(), y.data(), y.get_device_mem_stat());

  logger.func_out();
}
} // namespace

namespace vml {
void floor(const vector<double> &a, vector<double> &y) { vfloor_core(a, y); }

void floor(const vector<float> &a, vector<float> &y) { vfloor_core(a, y); }
} // namespace vml
} // namespace monolish
