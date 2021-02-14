#include "../../../include/monolish_blas.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {

namespace {
template <typename F1> double Dnrm1_core(const F1 &x) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  double ans = 0;
  const double *xd = x.data();
  size_t size = x.size();
  const size_t xoffset = x.get_offset();

  if (x.get_device_mem_stat() == true) {
#if MONOLISH_USE_GPU
#pragma omp target teams distribute parallel for reduction(+ : ans) map (tofrom: ans)
    for (size_t i = 0; i < size; i++) {
      ans += std::abs(xd[i + xoffset]);
    }
#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  } else {
#pragma omp parallel for reduction(+ : ans)
    for (size_t i = 0; i < size; i++) {
      ans += std::abs(xd[i + xoffset]);
    }
  }

  logger.func_out();
  return ans;
}

template <typename F1> float Snrm1_core(const F1 &x) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  float ans = 0;
  const float *xd = x.data();
  size_t size = x.size();
  const size_t xoffset = x.get_offset();

  if (x.get_device_mem_stat() == true) {
#if MONOLISH_USE_GPU
#pragma omp target teams distribute parallel for reduction(+ : ans) map (tofrom: ans)
    for (size_t i = 0; i < size; i++) {
      ans += std::abs(xd[i + xoffset]);
    }
#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  } else {
#pragma omp parallel for reduction(+ : ans)
    for (size_t i = 0; i < size; i++) {
      ans += std::abs(xd[i + xoffset]);
    }
  }

  logger.func_out();
  return ans;
}

} // namespace

namespace blas {

double nrm1(const vector<double> &x) { return Dnrm1_core(x); }
double nrm1(const view1D<vector<double>, double> &x) { return Dnrm1_core(x); }
float nrm1(const vector<float> &x) { return Snrm1_core(x); }
float nrm1(const view1D<vector<float>, float> &x) { return Snrm1_core(x); }

void nrm1(const vector<double> &x, double &ans) { ans = nrm1(x); }
void nrm1(const view1D<vector<double>, double> &x, double &ans) {
  ans = nrm1(x);
}
void nrm1(const vector<float> &x, float &ans) { ans = nrm1(x); }
void nrm1(const view1D<vector<float>, float> &x, float &ans) { ans = nrm1(x); }

} // namespace blas

} // namespace monolish
