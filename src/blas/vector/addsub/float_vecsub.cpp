#include "../../../../include/monolish_blas.hpp"
#include "../../../internal/monolish_internal.hpp"

namespace monolish {
// vecsub ///////////////////
//
namespace {
template <typename F1, typename F2, typename F3>
void vecsub_core(const F1 &a, const F2 &b, F3 &y) {
  monolish::Logger &logger = monolish::Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(monolish::util::is_same_size(a, b, y));
  assert(monolish::util::is_same_device_mem_stat(a, b, y));

  monolish::internal::vadd(y.size(), a.data(), b.data(), y.data(),
                           y.get_device_mem_stat());

  logger.func_out();
}
} // namespace

namespace blas {
void vecsub(const vector<float> &a, const vector<float> &b,
            vector<float> &y) {
  vecsub_core(a, b, y);
}
} // namespace blas

} // namespace monolish
