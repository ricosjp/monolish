#include "../../../include/monolish_vml.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {

namespace {
template <typename F1, typename F2, typename F3> 
  void svsub_core(const F1 &a, const F2 alpha, F3 &y) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(a, y));
  assert(util::is_same_device_mem_stat(a, y));

  internal::vsub(y.size(), a.data(), alpha, y.data(), y.get_device_mem_stat());

  logger.func_out();
}

} // namespace

namespace vml {
void sub(const vector<double> &a, const double alpha, vector<double> &y) { 
  svsub_core(a, alpha, y);
}

void sub(const vector<float> &a, const float alpha, vector<float> &y) { 
  svsub_core(a, alpha, y);
}

} // namespace blas

} // namespace monolish
