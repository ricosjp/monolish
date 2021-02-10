#include "../../../include/monolish_blas.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {
// vecsub ///////////////////
//
namespace {
template <typename F1, typename F2, typename F3>
void vecsub_core(const F1 &a, const F2 &b, F3 &y) {
  monolish::Logger &logger = monolish::Logger::get_instance();
  logger.func_in(monolish_func);

  const size_t aoffset = a.get_offset();
  const size_t boffset = b.get_offset();
  const size_t yoffset = y.get_offset();

  // err
  assert(monolish::util::is_same_size(a, b, y));
  assert(monolish::util::is_same_device_mem_stat(a, b, y));

  monolish::internal::vsub(y.size(), a.data()+aoffset, b.data()+boffset, y.data()+yoffset,
                           y.get_device_mem_stat());

  logger.func_out();
}
} // namespace

namespace blas {

void vecsub(const vector<double> &a, const vector<double> &b,
            vector<double> &y) {
  vecsub_core(a, b, y);
}
void vecsub(const vector<double> &a, const vector<double> &b,
            view1D<vector<double>, double> &y) {
  vecsub_core(a, b, y);
}
void vecsub(const vector<double> &a, const view1D<vector<double>, double> &b,
            vector<double> &y) {
  vecsub_core(a, b, y);
}
void vecsub(const vector<double> &a, const view1D<vector<double>, double> &b,
            view1D<vector<double>, double> &y) {
  vecsub_core(a, b, y);
}
void vecsub(const view1D<vector<double>, double> &a, const vector<double> &b,
            vector<double> &y) {
  vecsub_core(a, b, y);
}
void vecsub(const view1D<vector<double>, double> &a, const vector<double> &b,
            view1D<vector<double>, double> &y) {
  vecsub_core(a, b, y);
}
void vecsub(const view1D<vector<double>, double> &a,
            const view1D<vector<double>, double> &b, vector<double> &y) {
  vecsub_core(a, b, y);
}
void vecsub(const view1D<vector<double>, double> &a,
            const view1D<vector<double>, double> &b,
            view1D<vector<double>, double> &y) {
  vecsub_core(a, b, y);
}

void vecsub(const vector<float> &a, const vector<float> &b, vector<float> &y) {
  vecsub_core(a, b, y);
}
void vecsub(const vector<float> &a, const vector<float> &b,
            view1D<vector<float>, float> &y) {
  vecsub_core(a, b, y);
}
void vecsub(const vector<float> &a, const view1D<vector<float>, float> &b,
            vector<float> &y) {
  vecsub_core(a, b, y);
}
void vecsub(const vector<float> &a, const view1D<vector<float>, float> &b,
            view1D<vector<float>, float> &y) {
  vecsub_core(a, b, y);
}
void vecsub(const view1D<vector<float>, float> &a, const vector<float> &b,
            vector<float> &y) {
  vecsub_core(a, b, y);
}
void vecsub(const view1D<vector<float>, float> &a, const vector<float> &b,
            view1D<vector<float>, float> &y) {
  vecsub_core(a, b, y);
}
void vecsub(const view1D<vector<float>, float> &a,
            const view1D<vector<float>, float> &b, vector<float> &y) {
  vecsub_core(a, b, y);
}
void vecsub(const view1D<vector<float>, float> &a,
            const view1D<vector<float>, float> &b,
            view1D<vector<float>, float> &y) {
  vecsub_core(a, b, y);
}

} // namespace blas
} // namespace monolish
