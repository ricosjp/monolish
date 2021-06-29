#include "../../../include/monolish_blas.hpp"
#include "../monolish_internal.hpp"

namespace monolish {
namespace internal {

//////////////
// sqrt
//////////////
void vsqrt(const size_t N, const double *a, double *y, bool gpu_status) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  if (gpu_status == true) {
#if MONOLISH_USE_NVIDIA_GPU
#pragma omp target teams distribute parallel for
    for (size_t i = 0; i < N; i++) {
      y[i] = std::sqrt(a[i]);
    }
#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  } else {
#if MONOLISH_USE_MKL
    vdSqrt(N, a, y);
#else
#pragma omp parallel for
    for (size_t i = 0; i < N; i++) {
      y[i] = std::sqrt(a[i]);
    }
#endif
  }
  logger.func_out();
}

//////////////
// pow
//////////////
void vpow(const size_t N, const double *a, const double *b, double *y,
          bool gpu_status) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  if (gpu_status == true) {
#if MONOLISH_USE_NVIDIA_GPU
#pragma omp target teams distribute parallel for
    for (size_t i = 0; i < N; i++) {
      y[i] = std::pow(a[i], b[i]);
    }
#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  } else {
#if MONOLISH_USE_MKL
    vdPow(N, a, b, y);
#else
#pragma omp parallel for
    for (size_t i = 0; i < N; i++) {
      y[i] = std::pow(a[i], b[i]);
    }
#endif
  }
  logger.func_out();
}

void vpow(const size_t N, const double *a, const double alpha, double *y,
          bool gpu_status) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  if (gpu_status == true) {
#if MONOLISH_USE_NVIDIA_GPU
#pragma omp target teams distribute parallel for
    for (size_t i = 0; i < N; i++) {
      y[i] = std::pow(a[i], alpha);
    }
#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  } else {
#if MONOLISH_USE_MKL
    vdPowx(N, a, alpha, y);
#else
#pragma omp parallel for
    for (size_t i = 0; i < N; i++) {
      y[i] = std::pow(a[i], alpha);
    }
#endif
  }
  logger.func_out();
}

//////////////
// sin
//////////////
void vsin(const size_t N, const double *a, double *y, bool gpu_status) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  if (gpu_status == true) {
#if MONOLISH_USE_NVIDIA_GPU
#pragma omp target teams distribute parallel for
    for (size_t i = 0; i < N; i++) {
      y[i] = std::sin(a[i]);
    }
#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  } else {
#if MONOLISH_USE_MKL
    vdSin(N, a, y);
#else
#pragma omp parallel for
    for (size_t i = 0; i < N; i++) {
      y[i] = std::sin(a[i]);
    }
#endif
  }
  logger.func_out();
}

void vsinh(const size_t N, const double *a, double *y, bool gpu_status) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  if (gpu_status == true) {
#if MONOLISH_USE_NVIDIA_GPU
#pragma omp target teams distribute parallel for
    for (size_t i = 0; i < N; i++) {
      y[i] = std::sinh(a[i]);
    }
#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  } else {
#if MONOLISH_USE_MKL
    vdSinh(N, a, y);
#else
#pragma omp parallel for
    for (size_t i = 0; i < N; i++) {
      y[i] = std::sinh(a[i]);
    }
#endif
  }
  logger.func_out();
}

void vasin(const size_t N, const double *a, double *y, bool gpu_status) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  if (gpu_status == true) {
#if MONOLISH_USE_NVIDIA_GPU
#pragma omp target teams distribute parallel for
    for (size_t i = 0; i < N; i++) {
      y[i] = std::asin(a[i]);
    }
#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  } else {
#if MONOLISH_USE_MKL
    vdAsin(N, a, y);
#else
#pragma omp parallel for
    for (size_t i = 0; i < N; i++) {
      y[i] = std::asin(a[i]);
    }
#endif
  }
  logger.func_out();
}
void vasinh(const size_t N, const double *a, double *y, bool gpu_status) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  if (gpu_status == true) {
#if MONOLISH_USE_NVIDIA_GPU
#pragma omp target teams distribute parallel for
    for (size_t i = 0; i < N; i++) {
      y[i] = std::asinh(a[i]);
    }
#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  } else {
#if MONOLISH_USE_MKL
    vdAsinh(N, a, y);
#else
#pragma omp parallel for
    for (size_t i = 0; i < N; i++) {
      y[i] = std::asinh(a[i]);
    }
#endif
  }
  logger.func_out();
}

//////////////
// tan
//////////////
void vtan(const size_t N, const double *a, double *y, bool gpu_status) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  if (gpu_status == true) {
#if MONOLISH_USE_NVIDIA_GPU
#pragma omp target teams distribute parallel for
    for (size_t i = 0; i < N; i++) {
      y[i] = std::tan(a[i]);
    }
#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  } else {
#if MONOLISH_USE_MKL
    vdTan(N, a, y);
#else
#pragma omp parallel for
    for (size_t i = 0; i < N; i++) {
      y[i] = std::tan(a[i]);
    }
#endif
  }
  logger.func_out();
}

void vtanh(const size_t N, const double *a, double *y, bool gpu_status) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  if (gpu_status == true) {
#if MONOLISH_USE_NVIDIA_GPU
#pragma omp target teams distribute parallel for
    for (size_t i = 0; i < N; i++) {
      y[i] = std::tanh(a[i]);
    }
#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  } else {
#if MONOLISH_USE_MKL
    vdTanh(N, a, y);
#else
#pragma omp parallel for
    for (size_t i = 0; i < N; i++) {
      y[i] = std::tanh(a[i]);
    }
#endif
  }
  logger.func_out();
}

void vatan(const size_t N, const double *a, double *y, bool gpu_status) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  if (gpu_status == true) {
#if MONOLISH_USE_NVIDIA_GPU
#pragma omp target teams distribute parallel for
    for (size_t i = 0; i < N; i++) {
      y[i] = std::atan(a[i]);
    }
#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  } else {
#if MONOLISH_USE_MKL
    vdAtan(N, a, y);
#else
#pragma omp parallel for
    for (size_t i = 0; i < N; i++) {
      y[i] = std::atan(a[i]);
    }
#endif
  }
  logger.func_out();
}
void vatanh(const size_t N, const double *a, double *y, bool gpu_status) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  if (gpu_status == true) {
#if MONOLISH_USE_NVIDIA_GPU
#pragma omp target teams distribute parallel for
    for (size_t i = 0; i < N; i++) {
      y[i] = std::atanh(a[i]);
    }
#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  } else {
#if MONOLISH_USE_MKL
    vdAtanh(N, a, y);
#else
#pragma omp parallel for
    for (size_t i = 0; i < N; i++) {
      y[i] = std::atanh(a[i]);
    }
#endif
  }
  logger.func_out();
}

//////////////
// ceil, floor
//////////////
void vceil(const size_t N, const double *a, double *y, bool gpu_status) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  if (gpu_status == true) {
#if MONOLISH_USE_NVIDIA_GPU
#pragma omp target teams distribute parallel for
    for (size_t i = 0; i < N; i++) {
      y[i] = std::ceil(a[i]);
    }
#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  } else {
#if MONOLISH_USE_MKL
    vdCeil(N, a, y);
#else
#pragma omp parallel for
    for (size_t i = 0; i < N; i++) {
      y[i] = std::ceil(a[i]);
    }
#endif
  }
  logger.func_out();
}

void vfloor(const size_t N, const double *a, double *y, bool gpu_status) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  if (gpu_status == true) {
#if MONOLISH_USE_NVIDIA_GPU
#pragma omp target teams distribute parallel for
    for (size_t i = 0; i < N; i++) {
      y[i] = std::floor(a[i]);
    }
#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  } else {
#if MONOLISH_USE_MKL
    vdFloor(N, a, y);
#else
#pragma omp parallel for
    for (size_t i = 0; i < N; i++) {
      y[i] = std::floor(a[i]);
    }
#endif
  }
  logger.func_out();
}

//////////////
// sign
//////////////
void vsign(const size_t N, const double *a, double *y, bool gpu_status) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  if (gpu_status == true) {
#if MONOLISH_USE_NVIDIA_GPU
#pragma omp target teams distribute parallel for
    for (size_t i = 0; i < N; i++) {
      y[i] = -1 * a[i];
    }
#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  } else {
#pragma omp parallel for
    for (size_t i = 0; i < N; i++) {
      y[i] = -1 * a[i];
    }
  }
  logger.func_out();
}
//////////////
// max
//////////////
double vmax(const size_t N, const double *y, bool gpu_status) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  double max = y[0];

  if (gpu_status == true) {
#if MONOLISH_USE_NVIDIA_GPU
#pragma omp target teams distribute parallel for reduction(max : max)
    for (size_t i = 0; i < N; i++) {
      if (y[i] > max) {
        max = y[i];
      }
    }
#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  } else {
#pragma omp parallel for reduction(max : max)
    for (size_t i = 0; i < N; i++) {
      if (y[i] > max) {
        max = y[i];
      }
    }
  }
  logger.func_out();
  return max;
}

void vmax(const size_t N, const double *a, const double *b, double *y,
          bool gpu_status) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  if (gpu_status == true) {
#if MONOLISH_USE_NVIDIA_GPU
#pragma omp target teams distribute parallel for
    for (size_t i = 0; i < N; i++) {
      y[i] = std::max(a[i], b[i]);
    }
#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  } else {
#pragma omp parallel for
    for (size_t i = 0; i < N; i++) {
      y[i] = std::max(a[i], b[i]);
    }
  }
  logger.func_out();
}

//////////////
// min
//////////////
double vmin(const size_t N, const double *y, bool gpu_status) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  double min = y[0];

  if (gpu_status == true) {
#if MONOLISH_USE_NVIDIA_GPU
#pragma omp target teams distribute parallel for reduction(min : min)
    for (size_t i = 0; i < N; i++) {
      if (y[i] < min) {
        min = y[i];
      }
    }
#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  } else {
#pragma omp parallel for reduction(min : min)
    for (size_t i = 0; i < N; i++) {
      if (y[i] < min) {
        min = y[i];
      }
    }
  }
  logger.func_out();
  return min;
}
void vmin(const size_t N, const double *a, const double *b, double *y,
          bool gpu_status) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  if (gpu_status == true) {
#if MONOLISH_USE_NVIDIA_GPU
#pragma omp target teams distribute parallel for
    for (size_t i = 0; i < N; i++) {
      y[i] = std::min(a[i], b[i]);
    }
#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  } else {
#pragma omp parallel for
    for (size_t i = 0; i < N; i++) {
      y[i] = std::min(a[i], b[i]);
    }
  }
  logger.func_out();
}

} // namespace internal
} // namespace monolish
