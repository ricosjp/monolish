#include "../../../include/monolish_blas.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {
namespace matrix {

// diag
template <typename T> void CRS<T>::diag(vector<T> &vec) const {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  size_t n = get_row() < get_col() ? rowN : colN;
  T *vecd = vec.data();

  assert(n == vec.size());
  assert(get_device_mem_stat() == vec.get_device_mem_stat());

  const T *vald = val.data();
  const int *rowd = row_ptr.data();
  const int *cold = col_ind.data();

  if (gpu_status == true) {
#if MONOLISH_USE_NVIDIA_GPU // gpu
#pragma omp target teams distribute parallel for
    for (size_t i = 0; i < n; i++) {
      vecd[i] = 0;
    }
#pragma omp target teams distribute parallel for
    for (size_t i = 0; i < n; i++) {
      for (int j = rowd[i]; j < rowd[i + 1]; j++) {
        if ((int)i == cold[j]) {
          vecd[i] = vald[j];
        }
      }
    }
#else
    throw std::runtime_error("error USE_GPU is false, but gpu_status == true");
#endif
  } else {
#pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
      vecd[i] = 0;
    }
#pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
      for (int j = rowd[i]; j < rowd[i + 1]; j++) {
        if ((int)i == cold[j]) {
          vecd[i] = vald[j];
        }
      }
    }
  }

  logger.func_out();
}
template void monolish::matrix::CRS<double>::diag(vector<double> &vec) const;
template void monolish::matrix::CRS<float>::diag(vector<float> &vec) const;

template <typename T> void CRS<T>::diag(view1D<vector<T>, T> &vec) const {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  size_t n = get_row() < get_col() ? rowN : colN;
  T *vecd = vec.data();

  assert(n == vec.size());
  assert(get_device_mem_stat() == vec.get_device_mem_stat());

  const T *vald = val.data();
  const int *rowd = row_ptr.data();
  const int *cold = col_ind.data();

  if (gpu_status == true) {
#if MONOLISH_USE_NVIDIA_GPU // gpu
#pragma omp target teams distribute parallel for
    for (size_t i = 0; i < n; i++) {
      vecd[i] = 0;
    }
#pragma omp target teams distribute parallel for
    for (size_t i = 0; i < n; i++) {
      for (int j = rowd[i]; j < rowd[i + 1]; j++) {
        if ((int)i == cold[j]) {
          vecd[i] = vald[j];
        }
      }
    }
#else
    throw std::runtime_error("error USE_GPU is false, but gpu_status == true");
#endif
  } else {
#pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
      vecd[i] = 0;
    }
#pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
      for (int j = rowd[i]; j < rowd[i + 1]; j++) {
        if ((int)i == cold[j]) {
          vecd[i] = vald[j];
        }
      }
    }
  }

  logger.func_out();
}
template void
monolish::matrix::CRS<double>::diag(view1D<vector<double>, double> &vec) const;
template void
monolish::matrix::CRS<float>::diag(view1D<vector<float>, float> &vec) const;

template <typename T>
void CRS<T>::diag(view1D<matrix::Dense<T>, T> &vec) const {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  size_t n = get_row() < get_col() ? rowN : colN;
  T *vecd = vec.data();

  assert(n == vec.size());
  assert(get_device_mem_stat() == vec.get_device_mem_stat());

  const T *vald = val.data();
  const int *rowd = row_ptr.data();
  const int *cold = col_ind.data();

  if (gpu_status == true) {
#if MONOLISH_USE_NVIDIA_GPU // gpu
#pragma omp target teams distribute parallel for
    for (size_t i = 0; i < n; i++) {
      vecd[i] = 0;
    }
#pragma omp target teams distribute parallel for
    for (size_t i = 0; i < n; i++) {
      for (int j = rowd[i]; j < rowd[i + 1]; j++) {
        if ((int)i == cold[j]) {
          vecd[i] = vald[j];
        }
      }
    }
#else
    throw std::runtime_error("error USE_GPU is false, but gpu_status == true");
#endif
  } else {
#pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
      vecd[i] = 0;
    }
#pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
      for (int j = rowd[i]; j < rowd[i + 1]; j++) {
        if ((int)i == cold[j]) {
          vecd[i] = vald[j];
        }
      }
    }
  }

  logger.func_out();
}
template void monolish::matrix::CRS<double>::diag(
    view1D<matrix::Dense<double>, double> &vec) const;
template void monolish::matrix::CRS<float>::diag(
    view1D<matrix::Dense<float>, float> &vec) const;

// row
template <typename T> void CRS<T>::row(const size_t r, vector<T> &vec) const {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  size_t n = get_row();
  T *vecd = vec.data();

  const T *vald = val.data();
  const int *rowd = row_ptr.data();

  assert(n == vec.size());
  assert(get_device_mem_stat() == vec.get_device_mem_stat());

  if (gpu_status == true) {
#if MONOLISH_USE_NVIDIA_GPU // gpu
    const int *indexd = col_ind.data();

#pragma omp target teams distribute parallel for
    for (size_t i = 0; i < n; i++) {
      vecd[i] = 0;
    }
#pragma omp target teams distribute parallel for
    for (int j = rowd[r]; j < rowd[r + 1]; j++) {
      vecd[indexd[j]] = vald[j];
    }
#else
    throw std::runtime_error("error USE_GPU is false, but gpu_status == true");
#endif
  } else {
#pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
      vecd[i] = 0;
    }
#pragma omp parallel for
    for (int j = rowd[r]; j < rowd[r + 1]; j++) {
      vecd[col_ind[j]] = vald[j];
    }
  }

  logger.func_out();
}
template void monolish::matrix::CRS<double>::row(const size_t r,
                                                 vector<double> &vec) const;
template void monolish::matrix::CRS<float>::row(const size_t r,
                                                vector<float> &vec) const;

template <typename T>
void CRS<T>::row(const size_t r, view1D<vector<T>, T> &vec) const {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  size_t n = get_row();
  T *vecd = vec.data();

  const T *vald = val.data();
  const int *rowd = row_ptr.data();

  assert(n == vec.size());
  assert(get_device_mem_stat() == vec.get_device_mem_stat());

  if (gpu_status == true) {
#if MONOLISH_USE_NVIDIA_GPU // gpu
    const int *indexd = col_ind.data();

#pragma omp target teams distribute parallel for
    for (size_t i = 0; i < n; i++) {
      vecd[i] = 0;
    }
#pragma omp target teams distribute parallel for
    for (int j = rowd[r]; j < rowd[r + 1]; j++) {
      vecd[indexd[j]] = vald[j];
    }
#else
    throw std::runtime_error("error USE_GPU is false, but gpu_status == true");
#endif
  } else {
#pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
      vecd[i] = 0;
    }
#pragma omp parallel for
    for (int j = rowd[r]; j < rowd[r + 1]; j++) {
      vecd[col_ind[j]] = vald[j];
    }
  }

  logger.func_out();
}
template void
monolish::matrix::CRS<double>::row(const size_t r,
                                   view1D<vector<double>, double> &vec) const;
template void
monolish::matrix::CRS<float>::row(const size_t r,
                                  view1D<vector<float>, float> &vec) const;

template <typename T>
void CRS<T>::row(const size_t r, view1D<matrix::Dense<T>, T> &vec) const {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  size_t n = get_row();
  T *vecd = vec.data();

  const T *vald = val.data();
  const int *rowd = row_ptr.data();

  assert(n == vec.size());
  assert(get_device_mem_stat() == vec.get_device_mem_stat());

  if (gpu_status == true) {
#if MONOLISH_USE_NVIDIA_GPU // gpu
    const int *indexd = col_ind.data();

#pragma omp target teams distribute parallel for
    for (size_t i = 0; i < n; i++) {
      vecd[i] = 0;
    }
#pragma omp target teams distribute parallel for
    for (int j = rowd[r]; j < rowd[r + 1]; j++) {
      vecd[indexd[j]] = vald[j];
    }
#else
    throw std::runtime_error("error USE_GPU is false, but gpu_status == true");
#endif
  } else {
#pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
      vecd[i] = 0;
    }
#pragma omp parallel for
    for (int j = rowd[r]; j < rowd[r + 1]; j++) {
      vecd[col_ind[j]] = vald[j];
    }
  }

  logger.func_out();
}
template void monolish::matrix::CRS<double>::row(
    const size_t r, view1D<matrix::Dense<double>, double> &vec) const;
template void monolish::matrix::CRS<float>::row(
    const size_t r, view1D<matrix::Dense<float>, float> &vec) const;

// col
template <typename T> void CRS<T>::col(const size_t c, vector<T> &vec) const {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  size_t n = get_col();
  T *vecd = vec.data();

  const T *vald = val.data();
  const int *rowd = row_ptr.data();
  const int *cold = col_ind.data();

  assert(n == vec.size());
  assert(get_device_mem_stat() == vec.get_device_mem_stat());

  if (gpu_status == true) {
#if MONOLISH_USE_NVIDIA_GPU // gpu
#pragma omp target teams distribute parallel for
    for (size_t i = 0; i < n; i++) {
      vecd[i] = 0;
    }
#pragma omp target teams distribute parallel for
    for (size_t i = 0; i < n; i++) {
      for (int j = rowd[i]; j < rowd[i + 1]; j++) {
        if ((int)c == cold[j]) {
          vecd[i] = vald[j];
        }
      }
    }
#else
    throw std::runtime_error("error USE_GPU is false, but gpu_status == true");
#endif
  } else {
#pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
      vecd[i] = 0;
    }
#pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
      for (int j = rowd[i]; j < rowd[i + 1]; j++) {
        if ((int)c == cold[j]) {
          vecd[i] = vald[j];
        }
      }
    }
  }

  logger.func_out();
}
template void monolish::matrix::CRS<double>::col(const size_t c,
                                                 vector<double> &vec) const;
template void monolish::matrix::CRS<float>::col(const size_t c,
                                                vector<float> &vec) const;

template <typename T>
void CRS<T>::col(const size_t c, view1D<vector<T>, T> &vec) const {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  size_t n = get_col();
  T *vecd = vec.data();

  const T *vald = val.data();
  const int *rowd = row_ptr.data();
  const int *cold = col_ind.data();

  assert(n == vec.size());
  assert(get_device_mem_stat() == vec.get_device_mem_stat());

  if (gpu_status == true) {
#if MONOLISH_USE_NVIDIA_GPU // gpu
#pragma omp target teams distribute parallel for
    for (size_t i = 0; i < n; i++) {
      vecd[i] = 0;
    }
#pragma omp target teams distribute parallel for
    for (size_t i = 0; i < n; i++) {
      for (int j = rowd[i]; j < rowd[i + 1]; j++) {
        if ((int)c == cold[j]) {
          vecd[i] = vald[j];
        }
      }
    }
#else
    throw std::runtime_error("error USE_GPU is false, but gpu_status == true");
#endif
  } else {
#pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
      vecd[i] = 0;
    }
#pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
      for (int j = rowd[i]; j < rowd[i + 1]; j++) {
        if ((int)c == cold[j]) {
          vecd[i] = vald[j];
        }
      }
    }
  }

  logger.func_out();
}
template void
monolish::matrix::CRS<double>::col(const size_t c,
                                   view1D<vector<double>, double> &vec) const;
template void
monolish::matrix::CRS<float>::col(const size_t c,
                                  view1D<vector<float>, float> &vec) const;

template <typename T>
void CRS<T>::col(const size_t c, view1D<matrix::Dense<T>, T> &vec) const {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  size_t n = get_col();
  T *vecd = vec.data();

  const T *vald = val.data();
  const int *rowd = row_ptr.data();
  const int *cold = col_ind.data();

  assert(n == vec.size());
  assert(get_device_mem_stat() == vec.get_device_mem_stat());

  if (gpu_status == true) {
#if MONOLISH_USE_NVIDIA_GPU // gpu
#pragma omp target teams distribute parallel for
    for (size_t i = 0; i < n; i++) {
      vecd[i] = 0;
    }
#pragma omp target teams distribute parallel for
    for (size_t i = 0; i < n; i++) {
      for (int j = rowd[i]; j < rowd[i + 1]; j++) {
        if ((int)c == cold[j]) {
          vecd[i] = vald[j];
        }
      }
    }
#else
    throw std::runtime_error("error USE_GPU is false, but gpu_status == true");
#endif
  } else {
#pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
      vecd[i] = 0;
    }
#pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
      for (int j = rowd[i]; j < rowd[i + 1]; j++) {
        if ((int)c == cold[j]) {
          vecd[i] = vald[j];
        }
      }
    }
  }

  logger.func_out();
}
template void monolish::matrix::CRS<double>::col(
    const size_t c, view1D<matrix::Dense<double>, double> &vec) const;
template void monolish::matrix::CRS<float>::col(
    const size_t c, view1D<matrix::Dense<float>, float> &vec) const;

} // namespace matrix
} // namespace monolish
