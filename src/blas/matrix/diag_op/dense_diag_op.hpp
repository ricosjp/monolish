#include "../../../../include/monolish_blas.hpp"
#include "../../../internal/monolish_internal.hpp"

namespace monolish {
namespace {

// add scalar
template <typename T>
void Dense_diag_add_core(matrix::Dense<T> &MAT, const T alpha) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  T *vald = MAT.data();
  const auto N = MAT.get_col();
  const auto Len =
      MAT.get_row() < MAT.get_col() ? MAT.get_row() : MAT.get_col();

  if (MAT.get_device_mem_stat() == true) {
#if MONOLISH_USE_NVIDIA_GPU // gpu
#pragma omp target teams distribute parallel for
    for (auto i = decltype(Len){0}; i < Len; i++) {
      vald[N * i + i] += alpha;
    }
#else
    throw std::runtime_error("error USE_GPU is false, but gpu_status == true");
#endif
  } else {
#pragma omp parallel for
    for (auto i = decltype(Len){0}; i < Len; i++) {
      vald[N * i + i] += alpha;
    }
  }

  logger.func_out();
}

// add vector
template <typename T>
void Dense_diag_add_core(matrix::Dense<T> &MAT, const size_t size,
                         const T *vecd) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  T *vald = MAT.data();
  const auto N = MAT.get_col();
  const auto Len =
      MAT.get_row() < MAT.get_col() ? MAT.get_row() : MAT.get_col();

  assert(Len == size);

  if (MAT.get_device_mem_stat() == true) {
#if MONOLISH_USE_NVIDIA_GPU // gpu
#pragma omp target teams distribute parallel for
    for (auto i = decltype(Len){0}; i < Len; i++) {
      vald[N * i + i] += vecd[i];
    }
#else
    throw std::runtime_error("error USE_GPU is false, but gpu_status == true");
#endif
  } else {
#pragma omp parallel for
    for (auto i = decltype(Len){0}; i < Len; i++) {
      vald[N * i + i] += vecd[i];
    }
  }

  logger.func_out();
}

// sub scalar
template <typename T>
void Dense_diag_sub_core(matrix::Dense<T> &MAT, const T alpha) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  T *vald = MAT.data();
  const auto N = MAT.get_col();
  const auto Len =
      MAT.get_row() < MAT.get_col() ? MAT.get_row() : MAT.get_col();

  if (MAT.get_device_mem_stat() == true) {
#if MONOLISH_USE_NVIDIA_GPU // gpu
#pragma omp target teams distribute parallel for
    for (auto i = decltype(Len){0}; i < Len; i++) {
      vald[N * i + i] -= alpha;
    }
#else
    throw std::runtime_error("error USE_GPU is false, but gpu_status == true");
#endif
  } else {
#pragma omp parallel for
    for (auto i = decltype(Len){0}; i < Len; i++) {
      vald[N * i + i] -= alpha;
    }
  }

  logger.func_out();
}

// sub vector
template <typename T>
void Dense_diag_sub_core(matrix::Dense<T> &MAT, const size_t size,
                         const T *vecd) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  T *vald = MAT.data();
  const auto N = MAT.get_col();
  const auto Len =
      MAT.get_row() < MAT.get_col() ? MAT.get_row() : MAT.get_col();

  assert(Len == size);

  if (MAT.get_device_mem_stat() == true) {
#if MONOLISH_USE_NVIDIA_GPU // gpu
#pragma omp target teams distribute parallel for
    for (auto i = decltype(Len){0}; i < Len; i++) {
      vald[N * i + i] -= vecd[i];
    }
#else
    throw std::runtime_error("error USE_GPU is false, but gpu_status == true");
#endif
  } else {
#pragma omp parallel for
    for (auto i = decltype(Len){0}; i < Len; i++) {
      vald[N * i + i] -= vecd[i];
    }
  }

  logger.func_out();
}

// mul scalar
template <typename T>
void Dense_diag_mul_core(matrix::Dense<T> &MAT, const T alpha) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  T *vald = MAT.data();
  const auto N = MAT.get_col();
  const auto Len =
      MAT.get_row() < MAT.get_col() ? MAT.get_row() : MAT.get_col();

  if (MAT.get_device_mem_stat() == true) {
#if MONOLISH_USE_NVIDIA_GPU // gpu
#pragma omp target teams distribute parallel for
    for (auto i = decltype(Len){0}; i < Len; i++) {
      vald[N * i + i] *= alpha;
    }
#else
    throw std::runtime_error("error USE_GPU is false, but gpu_status == true");
#endif
  } else {
#pragma omp parallel for
    for (auto i = decltype(Len){0}; i < Len; i++) {
      vald[N * i + i] *= alpha;
    }
  }

  logger.func_out();
}

// mul vector
template <typename T>
void Dense_diag_mul_core(matrix::Dense<T> &MAT, const size_t size,
                         const T *vecd) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  T *vald = MAT.data();
  const auto N = MAT.get_col();
  const auto Len =
      MAT.get_row() < MAT.get_col() ? MAT.get_row() : MAT.get_col();

  assert(Len == size);

  if (MAT.get_device_mem_stat() == true) {
#if MONOLISH_USE_NVIDIA_GPU // gpu
#pragma omp target teams distribute parallel for
    for (auto i = decltype(Len){0}; i < Len; i++) {
      vald[N * i + i] *= vecd[i];
    }
#else
    throw std::runtime_error("error USE_GPU is false, but gpu_status == true");
#endif
  } else {
#pragma omp parallel for
    for (auto i = decltype(Len){0}; i < Len; i++) {
      vald[N * i + i] *= vecd[i];
    }
  }

  logger.func_out();
}

// div scalar
template <typename T>
void Dense_diag_div_core(matrix::Dense<T> &MAT, const T alpha) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  T *vald = MAT.data();
  const auto N = MAT.get_col();
  const auto Len =
      MAT.get_row() < MAT.get_col() ? MAT.get_row() : MAT.get_col();

  if (MAT.get_device_mem_stat() == true) {
#if MONOLISH_USE_NVIDIA_GPU // gpu
#pragma omp target teams distribute parallel for
    for (auto i = decltype(Len){0}; i < Len; i++) {
      vald[N * i + i] /= alpha;
    }
#else
    throw std::runtime_error("error USE_GPU is false, but gpu_status == true");
#endif
  } else {
#pragma omp parallel for
    for (auto i = decltype(Len){0}; i < Len; i++) {
      vald[N * i + i] /= alpha;
    }
  }

  logger.func_out();
}

// div vector
template <typename T>
void Dense_diag_div_core(matrix::Dense<T> &MAT, const size_t size,
                         const T *vecd) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  T *vald = MAT.data();
  const auto N = MAT.get_col();
  const auto Len =
      MAT.get_row() < MAT.get_col() ? MAT.get_row() : MAT.get_col();

  assert(Len == size);

  if (MAT.get_device_mem_stat() == true) {
#if MONOLISH_USE_NVIDIA_GPU // gpu
#pragma omp target teams distribute parallel for
    for (auto i = decltype(Len){0}; i < Len; i++) {
      vald[N * i + i] /= vecd[i];
    }
#else
    throw std::runtime_error("error USE_GPU is false, but gpu_status == true");
#endif
  } else {
#pragma omp parallel for
    for (auto i = decltype(Len){0}; i < Len; i++) {
      vald[N * i + i] /= vecd[i];
    }
  }

  logger.func_out();
}

} // namespace
} // namespace monolish
