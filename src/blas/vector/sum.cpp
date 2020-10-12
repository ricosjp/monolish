#include "../../../include/monolish_blas.hpp"
#include "../../monolish_internal.hpp"

#ifdef MONOLISH_USE_GPU
#include <cublas_v2.h>
#else
#include <cblas.h>
#endif

namespace monolish {

// double ///////////////////
double blas::sum(const vector<double> &x) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  double ans = 0;
  const double *xd = x.data();
  size_t size = x.size();

#if MONOLISH_USE_GPU
#pragma omp target teams distribute parallel for reduction(+ : ans) map (tofrom: ans)
  for (size_t i = 0; i < size; i++) {
    ans += xd[i];
  }
#else
#pragma omp parallel for reduction(+ : ans)
  for (size_t i = 0; i < size; i++) {
    ans += xd[i];
  }
#endif

  logger.func_out();
  return ans;
}

// float ///////////////////
float blas::sum(const vector<float> &x) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  float ans = 0;
  const float *xd = x.data();
  size_t size = x.size();

#if MONOLISH_USE_GPU
#pragma omp target teams distribute parallel for reduction(+ : ans) map (tofrom: ans)
  for (size_t i = 0; i < size; i++) {
    ans += xd[i];
  }
#else
#pragma omp parallel for reduction(+ : ans)
  for (size_t i = 0; i < size; i++) {
    ans += xd[i];
  }
#endif

  logger.func_out();
  return ans;
}

// void ///////////////////
void blas::sum(const vector<double> &x, double &ans) { ans = sum(x); }
void blas::sum(const vector<float> &x, float &ans) { ans = sum(x); }

} // namespace monolish
