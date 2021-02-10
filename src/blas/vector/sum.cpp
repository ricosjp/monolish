#include "../../../include/monolish_blas.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {

namespace {
template <typename F1> double Dsum_core(const F1 &x) {
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
      ans += xd[i+xoffset];
    }
#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  } else {
#pragma omp parallel for reduction(+ : ans)
    for (size_t i = 0; i < size; i++) {
      ans += xd[i+xoffset];
    }
  }

  logger.func_out();
  return ans;
}

template <typename F1> float Ssum_core(const F1 &x) {
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
      ans += xd[i+xoffset];
    }
#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  } else {
#pragma omp parallel for reduction(+ : ans)
    for (size_t i = 0; i < size; i++) {
      ans += xd[i+xoffset];
    }
  }

  logger.func_out();
  return ans;
}

} // namespace

namespace blas {

double sum(const vector<double> &x) { return Dsum_core(x); }
double sum(const view1D<vector<double>, double> &x) { return Dsum_core(x); }
void sum(const vector<double> &x, double &ans) { ans = sum(x); }
void sum(const view1D<vector<double>, double> &x, double &ans) { ans = sum(x); }

float sum(const vector<float> &x) { return Ssum_core(x); }
float sum(const view1D<vector<float>, float> &x) { return Ssum_core(x); }
void sum(const vector<float> &x, float &ans) { ans = sum(x); }
void sum(const view1D<vector<float>, float> &x, float &ans) { ans = sum(x); }

} // namespace blas

} // namespace monolish
