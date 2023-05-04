#pragma once

namespace monolish {

namespace {
template <typename F1, typename F2, typename F3, typename F4>
void Daxpyz_core(const F1 alpha, const F2 &x, const F3 &y, F4 &z) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(x, y, z));
  assert(util::is_same_device_mem_stat(x, y, z));

  const double *xd = x.begin();
  const double *yd = y.begin();
  double *zd = z.begin();
  auto size = x.size();

  if (x.get_device_mem_stat() == true) {
#if MONOLISH_USE_NVIDIA_GPU
#pragma omp target teams distribute parallel for
    for (auto i = decltype(size){0}; i < size; i++) {
      zd[i] = alpha * xd[i] + yd[i];
    }
#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  } else {
#pragma omp parallel for
    for (auto i = decltype(size){0}; i < size; i++) {
      zd[i] = alpha * xd[i] + yd[i];
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

  const float *xd = x.begin();
  const float *yd = y.begin();
  float *zd = z.begin();
  auto size = x.size();

  if (x.get_device_mem_stat() == true) {
#if MONOLISH_USE_NVIDIA_GPU
#pragma omp target teams distribute parallel for
    for (auto i = decltype(size){0}; i < size; i++) {
      zd[i] = alpha * xd[i] + yd[i];
    }
#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  } else {
#pragma omp parallel for
    for (auto i = decltype(size){0}; i < size; i++) {
      zd[i] = alpha * xd[i] + yd[i];
    }
  }
  logger.func_out();
}

} // namespace

} // namespace monolish
