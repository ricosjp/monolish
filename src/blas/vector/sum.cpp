#include "../../../include/monolish_blas.hpp"
#include "../../monolish_internal.hpp"

#ifdef USE_GPU
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

#if USE_GPU
#pragma acc data present(xd [0:size])
#pragma acc parallel
#pragma acc loop reduction(+ : ans)
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

#if USE_GPU
#pragma acc data present(xd [0:size])
#pragma acc parallel
#pragma acc loop reduction(+ : ans)
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
