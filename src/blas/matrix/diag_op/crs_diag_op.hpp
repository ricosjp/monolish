#include "../../../../include/monolish_blas.hpp"
#include "../../../internal/monolish_internal.hpp"

namespace monolish {
namespace {

// add scalar
template <typename T>
void CRS_diag_add_core(matrix::CRS<T> &MAT, const T alpha) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  T *vald = MAT.data();
  const auto *rowd = MAT.row_ptr.data();
  const auto *cold = MAT.col_ind.data();
  const auto rowN = MAT.get_row();

  if (MAT.get_device_mem_stat() == true) {
#if MONOLISH_USE_NVIDIA_GPU // gpu
#pragma omp target teams distribute parallel for
    for (auto i = decltype(rowN){0}; i < rowN; i++) {
      for (auto j = rowd[i]; j < rowd[i + 1]; j++) {
        if ((int)i == cold[j]) {
          vald[j] += alpha;
        }
      }
    }
#else
    throw std::runtime_error("error USE_GPU is false, but gpu_status == true");
#endif
  } else {
#pragma omp parallel for
    for (auto i = decltype(rowN){0}; i < rowN; i++) {
      for (auto j = rowd[i]; j < rowd[i + 1]; j++) {
        if ((int)i == cold[j]) {
          vald[j] += alpha;
        }
      }
    }
  }

  logger.func_out();
}

// add vector
template <typename T>
void CRS_diag_add_core(matrix::CRS<T> &MAT, const size_t size, const T *vecd) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  T *vald = MAT.data();
  const auto *rowd = MAT.row_ptr.data();
  const auto *cold = MAT.col_ind.data();

  const auto Len =
      MAT.get_row() < MAT.get_col() ? MAT.get_row() : MAT.get_col();
  assert(Len == size);

  if (MAT.get_device_mem_stat() == true) {
#if MONOLISH_USE_NVIDIA_GPU // gpu
#pragma omp target teams distribute parallel for
    for (auto i = decltype(Len){0}; i < Len; i++) {
      for (auto j = rowd[i]; j < rowd[i + 1]; j++) {
        if ((int)i == cold[j]) {
          vald[j] += vecd[i];
        }
      }
    }
#else
    throw std::runtime_error("error USE_GPU is false, but gpu_status == true");
#endif
  } else {
#pragma omp parallel for
    for (auto i = decltype(Len){0}; i < Len; i++) {
      for (auto j = rowd[i]; j < rowd[i + 1]; j++) {
        if ((int)i == cold[j]) {
          vald[j] += vecd[i];
        }
      }
    }
  }

  logger.func_out();
}

// sub scalar
template <typename T>
void CRS_diag_sub_core(matrix::CRS<T> &MAT, const T alpha) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  T *vald = MAT.data();
  const auto *rowd = MAT.row_ptr.data();
  const auto *cold = MAT.col_ind.data();
  const auto rowN = MAT.get_row();

  if (MAT.get_device_mem_stat() == true) {
#if MONOLISH_USE_NVIDIA_GPU // gpu
#pragma omp target teams distribute parallel for
    for (auto i = decltype(rowN){0}; i < rowN; i++) {
      for (auto j = rowd[i]; j < rowd[i + 1]; j++) {
        if ((int)i == cold[j]) {
          vald[j] -= alpha;
        }
      }
    }
#else
    throw std::runtime_error("error USE_GPU is false, but gpu_status == true");
#endif
  } else {
#pragma omp parallel for
    for (auto i = decltype(rowN){0}; i < rowN; i++) {
      for (auto j = rowd[i]; j < rowd[i + 1]; j++) {
        if ((int)i == cold[j]) {
          vald[j] -= alpha;
        }
      }
    }
  }

  logger.func_out();
}

// sub vector
template <typename T>
void CRS_diag_sub_core(matrix::CRS<T> &MAT, const size_t size, const T *vecd) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  T *vald = MAT.data();
  const auto *rowd = MAT.row_ptr.data();
  const auto *cold = MAT.col_ind.data();

  const auto Len =
      MAT.get_row() < MAT.get_col() ? MAT.get_row() : MAT.get_col();
  assert(Len == size);

  if (MAT.get_device_mem_stat() == true) {
#if MONOLISH_USE_NVIDIA_GPU // gpu
#pragma omp target teams distribute parallel for
    for (auto i = decltype(Len){0}; i < Len; i++) {
      for (auto j = rowd[i]; j < rowd[i + 1]; j++) {
        if ((int)i == cold[j]) {
          vald[j] -= vecd[i];
        }
      }
    }
#else
    throw std::runtime_error("error USE_GPU is false, but gpu_status == true");
#endif
  } else {
#pragma omp parallel for
    for (auto i = decltype(Len){0}; i < Len; i++) {
      for (auto j = rowd[i]; j < rowd[i + 1]; j++) {
        if ((int)i == cold[j]) {
          vald[j] -= vecd[i];
        }
      }
    }
  }

  logger.func_out();
}

// mul scalar
template <typename T>
void CRS_diag_mul_core(matrix::CRS<T> &MAT, const T alpha) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  T *vald = MAT.data();
  const auto *rowd = MAT.row_ptr.data();
  const auto *cold = MAT.col_ind.data();
  const auto rowN = MAT.get_row();

  if (MAT.get_device_mem_stat() == true) {
#if MONOLISH_USE_NVIDIA_GPU // gpu
#pragma omp target teams distribute parallel for
    for (auto i = decltype(rowN){0}; i < rowN; i++) {
      for (auto j = rowd[i]; j < rowd[i + 1]; j++) {
        if ((int)i == cold[j]) {
          vald[j] *= alpha;
        }
      }
    }
#else
    throw std::runtime_error("error USE_GPU is false, but gpu_status == true");
#endif
  } else {
#pragma omp parallel for
    for (auto i = decltype(rowN){0}; i < rowN; i++) {
      for (auto j = rowd[i]; j < rowd[i + 1]; j++) {
        if ((int)i == cold[j]) {
          vald[j] *= alpha;
        }
      }
    }
  }

  logger.func_out();
}

// mul vector
template <typename T>
void CRS_diag_mul_core(matrix::CRS<T> &MAT, const size_t size, const T *vecd) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  T *vald = MAT.data();
  const auto *rowd = MAT.row_ptr.data();
  const auto *cold = MAT.col_ind.data();

  const auto Len =
      MAT.get_row() < MAT.get_col() ? MAT.get_row() : MAT.get_col();
  assert(Len == size);

  if (MAT.get_device_mem_stat() == true) {
#if MONOLISH_USE_NVIDIA_GPU // gpu
#pragma omp target teams distribute parallel for
    for (auto i = decltype(Len){0}; i < Len; i++) {
      for (auto j = rowd[i]; j < rowd[i + 1]; j++) {
        if ((int)i == cold[j]) {
          vald[j] *= vecd[i];
        }
      }
    }
#else
    throw std::runtime_error("error USE_GPU is false, but gpu_status == true");
#endif
  } else {
#pragma omp parallel for
    for (auto i = decltype(Len){0}; i < Len; i++) {
      for (auto j = rowd[i]; j < rowd[i + 1]; j++) {
        if ((int)i == cold[j]) {
          vald[j] *= vecd[i];
        }
      }
    }
  }

  logger.func_out();
}

// div scalar
template <typename T>
void CRS_diag_div_core(matrix::CRS<T> &MAT, const T alpha) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  T *vald = MAT.data();
  const auto *rowd = MAT.row_ptr.data();
  const auto *cold = MAT.col_ind.data();
  const auto rowN = MAT.get_row();

  if (MAT.get_device_mem_stat() == true) {
#if MONOLISH_USE_NVIDIA_GPU // gpu
#pragma omp target teams distribute parallel for
    for (auto i = decltype(rowN){0}; i < rowN; i++) {
      for (auto j = rowd[i]; j < rowd[i + 1]; j++) {
        if ((int)i == cold[j]) {
          vald[j] /= alpha;
        }
      }
    }
#else
    throw std::runtime_error("error USE_GPU is false, but gpu_status == true");
#endif
  } else {
#pragma omp parallel for
    for (auto i = decltype(rowN){0}; i < rowN; i++) {
      for (auto j = rowd[i]; j < rowd[i + 1]; j++) {
        if ((int)i == cold[j]) {
          vald[j] /= alpha;
        }
      }
    }
  }

  logger.func_out();
}

// div vector
template <typename T>
void CRS_diag_div_core(matrix::CRS<T> &MAT, const size_t size, const T *vecd) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  T *vald = MAT.data();
  const auto *rowd = MAT.row_ptr.data();
  const auto *cold = MAT.col_ind.data();

  const auto Len =
      MAT.get_row() < MAT.get_col() ? MAT.get_row() : MAT.get_col();
  assert(Len == size);

  if (MAT.get_device_mem_stat() == true) {
#if MONOLISH_USE_NVIDIA_GPU // gpu
#pragma omp target teams distribute parallel for
    for (auto i = decltype(Len){0}; i < Len; i++) {
      for (auto j = rowd[i]; j < rowd[i + 1]; j++) {
        if ((int)i == cold[j]) {
          vald[j] /= vecd[i];
        }
      }
    }
#else
    throw std::runtime_error("error USE_GPU is false, but gpu_status == true");
#endif
  } else {
#pragma omp parallel for
    for (auto i = decltype(Len){0}; i < Len; i++) {
      for (auto j = rowd[i]; j < rowd[i + 1]; j++) {
        if ((int)i == cold[j]) {
          vald[j] /= vecd[i];
        }
      }
    }
  }

  logger.func_out();
}

} // namespace
} // namespace monolish
