#include "../../../include/monolish_blas.hpp"
#include "../../../include/monolish_vml.hpp"
#include "../../internal/monolish_internal.hpp"

#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace monolish {

template <>
void view1D<monolish::vector<double>, double>::print_all(bool force_cpu) const {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  double *val = target.data();
  if (get_device_mem_stat() == true && force_cpu == false) {
#if MONOLISH_USE_NVIDIA_GPU
#pragma omp target
    for (size_t i = first; i < last; i++) {
      printf("%f\n", val[i]);
    }
#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  } else {
    for (size_t i = first; i < last; i++) {
      printf("%f\n", val[i]);
    }
  }
  logger.util_out();
}

template <>
void view1D<monolish::matrix::Dense<double>, double>::print_all(
    bool force_cpu) const {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  double *val = target.val.data();
  if (get_device_mem_stat() == true && force_cpu == false) {
#if MONOLISH_USE_NVIDIA_GPU
#pragma omp target
    for (size_t i = first; i < last; i++) {
      printf("%f\n", val[i]);
    }
#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  } else {
    for (size_t i = first; i < last; i++) {
      printf("%f\n", val[i]);
    }
  }
  logger.util_out();
}

template <>
void view1D<monolish::vector<float>, float>::print_all(bool force_cpu) const {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  float *val = target.data();
  if (get_device_mem_stat() == true && force_cpu == false) {
#if MONOLISH_USE_NVIDIA_GPU
#pragma omp target
    for (size_t i = first; i < last; i++) {
      printf("%f\n", val[i]);
    }
#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  } else {
    for (size_t i = first; i < last; i++) {
      printf("%f\n", val[i]);
    }
  }
  logger.util_out();
}

template <>
void view1D<monolish::matrix::Dense<float>, float>::print_all(
    bool force_cpu) const {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  float *val = target.val.data();
  if (get_device_mem_stat() == true && force_cpu == false) {
#if MONOLISH_USE_NVIDIA_GPU
#pragma omp target
    for (size_t i = first; i < last; i++) {
      printf("%f\n", val[i]);
    }
#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  } else {
    for (size_t i = first; i < last; i++) {
      printf("%f\n", val[i]);
    }
  }
  logger.util_out();
}

} // namespace monolish
