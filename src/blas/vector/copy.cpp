#include "../../../include/monolish_blas.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {

namespace {
template <typename F1, typename F2> void copy_core(const F1 &a, F2 &y) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  // err
  assert(util::is_same_size(a, y));
  assert(util::is_same_device_mem_stat(a, y));

  internal::vcopy(y.size(), a.data(), y.data(), y.get_device_mem_stat());

  logger.util_out();
}

} // namespace

namespace blas {

void copy(const vector<double> &a, vector<double> &y) { copy_core(a, y); }
void copy(const vector<float> &a, vector<float> &y) { copy_core(a, y); }

} // namespace blas
} // namespace monolish
