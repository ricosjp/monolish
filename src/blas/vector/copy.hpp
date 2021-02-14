#include "../../../include/monolish_blas.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {

namespace {
template <typename F1, typename F2> void copy_core(const F1 &x, F2 &y) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  const size_t xoffset = x.get_offset();
  const size_t yoffset = y.get_offset();

  // err
  assert(util::is_same_size(x, y));
  assert(util::is_same_device_mem_stat(x, y));

  internal::vcopy(y.size(), x.data() + xoffset, y.data() + yoffset,
                  y.get_device_mem_stat());

  logger.util_out();
}

} // namespace

namespace blas {

void copy(const vector<double> &x, vector<double> &y) { copy_core(x, y); }
void copy(const vector<double> &x, view1D<vector<double>, double> &y) {
  copy_core(x, y);
}
void copy(const vector<double> &x, view1D<matrix::Dense<double>, double> &y) {
  copy_core(x, y);
}
void copy(const view1D<vector<double>, double> &x, vector<double> &y) {
  copy_core(x, y);
}
void copy(const view1D<matrix::Dense<double>, double> &x, vector<double> &y) {
  copy_core(x, y);
}
void copy(const view1D<vector<double>, double> &x,
          view1D<vector<double>, double> &y) {
  copy_core(x, y);
}
void copy(const view1D<matrix::Dense<double>, double> &x,
          view1D<matrix::Dense<double>, double> &y) {
  copy_core(x, y);
}

void copy(const vector<float> &x, vector<float> &y) { copy_core(x, y); }
void copy(const vector<float> &x, view1D<vector<float>, float> &y) {
  copy_core(x, y);
}
void copy(const vector<float> &x, view1D<matrix::Dense<float>, float> &y) {
  copy_core(x, y);
}
void copy(const view1D<vector<float>, float> &x, vector<float> &y) {
  copy_core(x, y);
}
void copy(const view1D<matrix::Dense<float>, float> &x, vector<float> &y) {
  copy_core(x, y);
}
void copy(const view1D<vector<float>, float> &x,
          view1D<vector<float>, float> &y) {
  copy_core(x, y);
}
void copy(const view1D<matrix::Dense<float>, float> &x,
          view1D<matrix::Dense<float>, float> &y) {
  copy_core(x, y);
}

} // namespace blas
} // namespace monolish
