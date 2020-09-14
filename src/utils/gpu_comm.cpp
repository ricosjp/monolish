#include "../../include/monolish_blas.hpp"
#include "../monolish_internal.hpp"

namespace monolish {
// vec ///////////////////////////////////////

// send
template <typename T> void vector<T>::send() {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

#if USE_GPU
  T *d = val.data();
  size_t N = val.size();

#pragma acc enter data copyin(d [0:N])
  gpu_status = true;
#endif
  logger.util_out();
}

// recv
template <typename T> void vector<T>::recv() {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

#if USE_GPU
  if (gpu_status == true) {
    T *d = val.data();
    size_t N = val.size();
#pragma acc exit data copyout(d [0:N])
    gpu_status = false;
  }
#endif
  logger.util_out();
}

// nonfree_recv
template <typename T> void vector<T>::nonfree_recv() {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

#if USE_GPU
  if (gpu_status == true) {
    T *d = val.data();
    size_t N = val.size();
#pragma acc update host(d [0:N])
  }
#endif
  logger.util_out();
}

// device_free
template <typename T> void vector<T>::device_free() {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

#if USE_GPU
  if (gpu_status == true) {
    T *d = val.data();
    size_t N = val.size();
#pragma acc exit data delete (d [0:N])
    gpu_status = false;
  }
#endif
  logger.util_out();
}

template void vector<float>::send();
template void vector<double>::send();

template void vector<float>::recv();
template void vector<double>::recv();

template void vector<float>::nonfree_recv();
template void vector<double>::nonfree_recv();

template void vector<float>::device_free();
template void vector<double>::device_free();

// CRS ///////////////////////////////////
// send
template <typename T> void matrix::CRS<T>::send() {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

#if USE_GPU
  T *vald = val.data();
  int *cold = col_ind.data();
  int *rowd = row_ptr.data();
  size_t N = get_row();
  size_t nnz = get_nnz();

#pragma acc enter data copyin(vald [0:nnz], cold [0:nnz], rowd [0:N + 1])
  gpu_status = true;
#endif
  logger.util_out();
}

// recv
template <typename T> void matrix::CRS<T>::recv() {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

#if USE_GPU
  if (gpu_status == true) {
    T *vald = val.data();
    int *cold = col_ind.data();
    int *rowd = row_ptr.data();
    size_t N = get_row();
    size_t nnz = get_nnz();

#pragma acc exit data copyout(vald [0:nnz], cold [0:nnz], rowd [0:N + 1])
    gpu_status = false;
  }
#endif
  logger.util_out();
}

// nonfree_recv
template <typename T> void matrix::CRS<T>::nonfree_recv() {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

#if USE_GPU
  if (gpu_status == true) {
    T *vald = val.data();
    int *cold = col_ind.data();
    int *rowd = row_ptr.data();
    size_t N = get_row();
    size_t nnz = get_nnz();

#pragma acc update host(vald [0:nnz], cold [0:nnz], rowd [0:N + 1])
  }
#endif
  logger.util_out();
}

// device_free
template <typename T> void matrix::CRS<T>::device_free() {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

#if USE_GPU
  if (gpu_status == true) {
    T *vald = val.data();
    int *cold = col_ind.data();
    int *rowd = row_ptr.data();
    size_t N = get_row();
    size_t nnz = get_nnz();

#pragma acc exit data delete (vald [0:nnz], cold [0:nnz], rowd [0:N + 1])

    gpu_status = false;
  }
#endif
  logger.util_out();
}
template void matrix::CRS<float>::send();
template void matrix::CRS<double>::send();

template void matrix::CRS<float>::recv();
template void matrix::CRS<double>::recv();

template void matrix::CRS<float>::nonfree_recv();
template void matrix::CRS<double>::nonfree_recv();

template void matrix::CRS<float>::device_free();
template void matrix::CRS<double>::device_free();
// Dense ///////////////////////////////////
// send
template <typename T> void matrix::Dense<T>::send() {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

#if USE_GPU
  T *vald = val.data();
  size_t nnz = get_nnz();

#pragma acc enter data copyin(vald [0:nnz])
  gpu_status = true;
#endif
  logger.util_out();
}

// recv
template <typename T> void matrix::Dense<T>::recv() {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

#if USE_GPU
  if (gpu_status == true) {
    T *vald = val.data();
    size_t nnz = get_nnz();

#pragma acc exit data copyout(vald [0:nnz])
    gpu_status = false;
  }
#endif
  logger.util_out();
}

// nonfree_recv
template <typename T> void matrix::Dense<T>::nonfree_recv() {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

#if USE_GPU
  if (gpu_status == true) {
    T *vald = val.data();
    size_t nnz = get_nnz();

#pragma acc update host(vald [0:nnz])
  }
#endif
  logger.util_out();
}

// device_free
template <typename T> void matrix::Dense<T>::device_free() {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

#if USE_GPU
  if (gpu_status == true) {
    T *vald = val.data();
    size_t nnz = get_nnz();

#pragma acc exit data delete (vald [0:nnz])

    gpu_status = false;
  }
#endif
  logger.util_out();
}
template void matrix::Dense<float>::send();
template void matrix::Dense<double>::send();

template void matrix::Dense<float>::recv();
template void matrix::Dense<double>::recv();

template void matrix::Dense<float>::nonfree_recv();
template void matrix::Dense<double>::nonfree_recv();

template void matrix::Dense<float>::device_free();
template void matrix::Dense<double>::device_free();
} // namespace monolish
