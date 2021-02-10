#include "../../../include/monolish_blas.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {

namespace {
template <typename F1, typename F2, typename F3, typename F4>
void Daxpyz_core(const F1 alpha, const F2 &x, const F3 &y, F4 &z) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(x, y, z));
  assert(util::is_same_device_mem_stat(x, y, z));

  const double *xd = x.data();
  const double *yd = y.data();
  double *zd = z.data();
  size_t size = x.size();
  const size_t xoffset = x.get_offset();
  const size_t yoffset = y.get_offset();
  const size_t zoffset = z.get_offset();

  if (x.get_device_mem_stat() == true) {
#if MONOLISH_USE_GPU
#pragma omp target teams distribute parallel for
    for (size_t i = 0; i < size; i++) {
      zd[i + zoffset] = alpha * xd[i + xoffset] + yd[i + yoffset];
    }
#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  } else {
#pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
      zd[i + zoffset] = alpha * xd[i + xoffset] + yd[i + yoffset];
    }
  }
  logger.func_out();
}

template <typename F1, typename F2, typename F3, typename F4>
void Saxpyz_core(const F1 alpha, const F2 &x, const F3 &y, F4 &z) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(x, y, z));
  assert(util::is_same_device_mem_stat(x, y, z));

  const float *xd = x.data();
  const float *yd = y.data();
  float *zd = z.data();
  size_t size = x.size();
  const size_t xoffset = x.get_offset();
  const size_t yoffset = y.get_offset();
  const size_t zoffset = z.get_offset();

  if (x.get_device_mem_stat() == true) {
#if MONOLISH_USE_GPU
#pragma omp target teams distribute parallel for
    for (size_t i = 0; i < size; i++) {
      zd[i + zoffset] = alpha * xd[i + xoffset] + yd[i + yoffset];
    }
#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  } else {
#pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
      zd[i + zoffset] = alpha * xd[i + xoffset] + yd[i + yoffset];
    }
  }
  logger.func_out();
}

} // namespace

namespace blas {
void axpyz(const double alpha, const vector<double> &x, const vector<double> &y,
           vector<double> &z) {
  Daxpyz_core(alpha, x, y, z);
}
void axpyz(const double alpha, const vector<double> &x, const vector<double> &y,
           view1D<vector<double>, double> &z) {
  Daxpyz_core(alpha, x, y, z);
}
void axpyz(const double alpha, const vector<double> &x,
           const view1D<vector<double>, double> &y, vector<double> &z) {
  Daxpyz_core(alpha, x, y, z);
}
void axpyz(const double alpha, const vector<double> &x,
           const view1D<vector<double>, double> &y,
           view1D<vector<double>, double> &z) {
  Daxpyz_core(alpha, x, y, z);
}
void axpyz(const double alpha, const view1D<vector<double>, double> &x,
           const vector<double> &y, vector<double> &z) {
  Daxpyz_core(alpha, x, y, z);
}
void axpyz(const double alpha, const view1D<vector<double>, double> &x,
           const vector<double> &y, view1D<vector<double>, double> &z) {
  Daxpyz_core(alpha, x, y, z);
}
void axpyz(const double alpha, const view1D<vector<double>, double> &x,
           const view1D<vector<double>, double> &y, vector<double> &z) {
  Daxpyz_core(alpha, x, y, z);
}
void axpyz(const double alpha, const view1D<vector<double>, double> &x,
           const view1D<vector<double>, double> &y,
           view1D<vector<double>, double> &z) {
  Daxpyz_core(alpha, x, y, z);
}

void axpyz(const float alpha, const vector<float> &x, const vector<float> &y,
           vector<float> &z) {
  Saxpyz_core(alpha, x, y, z);
}
void axpyz(const float alpha, const vector<float> &x, const vector<float> &y,
           view1D<vector<float>, float> &z) {
  Saxpyz_core(alpha, x, y, z);
}
void axpyz(const float alpha, const vector<float> &x,
           const view1D<vector<float>, float> &y, vector<float> &z) {
  Saxpyz_core(alpha, x, y, z);
}
void axpyz(const float alpha, const vector<float> &x,
           const view1D<vector<float>, float> &y,
           view1D<vector<float>, float> &z) {
  Saxpyz_core(alpha, x, y, z);
}
void axpyz(const float alpha, const view1D<vector<float>, float> &x,
           const vector<float> &y, vector<float> &z) {
  Saxpyz_core(alpha, x, y, z);
}
void axpyz(const float alpha, const view1D<vector<float>, float> &x,
           const vector<float> &y, view1D<vector<float>, float> &z) {
  Saxpyz_core(alpha, x, y, z);
}
void axpyz(const float alpha, const view1D<vector<float>, float> &x,
           const view1D<vector<float>, float> &y, vector<float> &z) {
  Saxpyz_core(alpha, x, y, z);
}
void axpyz(const float alpha, const view1D<vector<float>, float> &x,
           const view1D<vector<float>, float> &y,
           view1D<vector<float>, float> &z) {
  Saxpyz_core(alpha, x, y, z);
}

} // namespace blas
} // namespace monolish
