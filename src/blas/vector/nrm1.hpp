#pragma once

namespace monolish {

namespace {
template <typename F1> double Dnrm1_core(const F1 &x) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  double ans = 0;
  const double *xd = x.data();
  auto size = x.size();
  const auto xoffset = x.get_offset();

  if (x.get_device_mem_stat() == true) {
#if MONOLISH_USE_NVIDIA_GPU
#pragma omp target teams distribute parallel for reduction(+ : ans) map (tofrom: ans)
    for (auto i = decltype(size){0}; i < size; i++) {
      ans += std::abs(xd[i + xoffset]);
    }
#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  } else {
#pragma omp parallel for reduction(+ : ans)
    for (auto i = decltype(size){0}; i < size; i++) {
      ans += std::abs(xd[i + xoffset]);
    }
  }

#if MONOLISH_USE_MPI
  mpi::comm &comm = mpi::comm::get_instance();
  ans = comm.Allreduce(ans);
#endif

  logger.func_out();
  return ans;
}

template <typename F1> float Snrm1_core(const F1 &x) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  float ans = 0;
  const float *xd = x.data();
  auto size = x.size();
  const auto xoffset = x.get_offset();

  if (x.get_device_mem_stat() == true) {
#if MONOLISH_USE_NVIDIA_GPU
#pragma omp target teams distribute parallel for reduction(+ : ans) map (tofrom: ans)
    for (auto i = decltype(size){0}; i < size; i++) {
      ans += std::abs(xd[i + xoffset]);
    }
#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  } else {
#pragma omp parallel for reduction(+ : ans)
    for (auto i = decltype(size){0}; i < size; i++) {
      ans += std::abs(xd[i + xoffset]);
    }
  }

#if MONOLISH_USE_MPI
  mpi::comm &comm = mpi::comm::get_instance();
  ans = comm.Allreduce(ans);
#endif

  logger.func_out();
  return ans;
}

} // namespace
} // namespace monolish
