#include "../../../include/monolish_vml.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {
namespace {
template <typename F1, typename F2, typename F3>
void vvmax_core(const F1 &a, const F2 &b, F3 &y) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(a, b, y));
  assert(util::is_same_device_mem_stat(a, b, y));

  internal::vmax(y.size(), a.data(), b.data(), y.data(),
                 y.get_device_mem_stat());

  logger.func_out();
}

template <typename F1, typename F2> F2 vmax_core(const F1 &y) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  F2 ret = internal::vmax(y.size(), y.data(), y.get_device_mem_stat());

  logger.func_out();
  return ret;
}
} // namespace

namespace vml {
void max(const vector<double> &a, const vector<double> &b, vector<double> &y) {
  vvmax_core(a, b, y);
}

void max(const vector<float> &a, const vector<float> &b, vector<float> &y) {
  vvmax_core(a, b, y);
}

double max(const vector<double> &y) {
  return vmax_core<vector<double>, double>(y);
}

float max(const vector<float> &y) { return vmax_core<vector<float>, float>(y); }
} // namespace vml
} // namespace monolish
