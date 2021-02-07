#include "../../../include/monolish_vml.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {
namespace {
template <typename F1, typename F2, typename F3>
void vvmin_core(const F1 &a, const F2 &b, F3 &y) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(a, b, y));
  assert(util::is_same_device_mem_stat(a, b, y));

  internal::vmin(y.size(), a.data(), b.data(), y.data(),
                 y.get_device_mem_stat());

  logger.func_out();
}

template <typename F1, typename F2> F2 vmin_core(const F1 &y) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  F2 ret = internal::vmin(y.size(), y.data(), y.get_device_mem_stat());

  logger.func_out();
  return ret;
}
} // namespace

namespace vml {
void min(const vector<double> &a, const vector<double> &b, vector<double> &y) {
  vvmin_core(a, b, y);
}

void min(const vector<float> &a, const vector<float> &b, vector<float> &y) {
  vvmin_core(a, b, y);
}

double min(const vector<double> &y) {
  return vmin_core<vector<double>, double>(y);
}

float min(const vector<float> &y) { return vmin_core<vector<float>, float>(y); }
} // namespace vml
} // namespace monolish
