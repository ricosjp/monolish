#include "../../../include/monolish_blas.hpp"
#include "../../monolish_internal.hpp"

#ifdef USE_GPU
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#endif

#ifdef USE_MKL
#include <mkl.h>
#else
#include <cblas.h>
#endif

namespace monolish {

void tanh_core(size_t N, double* vec){
#if USE_GPU
#pragma acc data present(vec[0:N])
#pragma acc host_data use_device(vec)
    {
        for (size_t i = 0; i < N; i++) {
            vec[i] = tanh(vec[i]);
        }
    }
#else
#if USE_MKL
   vdTanh(N, vec, vec);
#else
 #pragma omp parallel for
   for (size_t i = 0; i < N; i++) {
    vec[i] = std::tanh(vec[i]);
   }
#endif
#endif
}

void tanh_core(size_t N, float* vec){
#if USE_GPU
#pragma acc data present(vec[0:N])
#pragma acc host_data use_device(vec)
    {
        for (size_t i = 0; i < N; i++) {
            vec[i] = tanhf(vec[i]);
        }
    }
#else
#if USE_MKL
   vsTanh(N, vec, vec);
#else
 #pragma omp parallel for
   for (size_t i = 0; i < N; i++) {
    vec[i] = std::tanh(vec[i]);
   }
#endif
#endif
}

/////////

template <typename T> void vector<T>::tanh() {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  T *vald = val.data();
  tanh_core(get_nnz(), vald);

  logger.func_out();
}

template void vector<double>::tanh();
template void vector<float>::tanh();

/////////

template <typename T> void matrix::Dense<T>::tanh() {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  T *vald = val.data();
  tanh_core(get_nnz(), vald);

  logger.func_out();
}

template void matrix::Dense<double>::tanh();
template void matrix::Dense<float>::tanh();

/////////

template <typename T> void matrix::CRS<T>::tanh() {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  T *vald = val.data();
  tanh_core(get_nnz(), vald);

  logger.func_out();
}

template void matrix::CRS<double>::tanh();
template void matrix::CRS<float>::tanh();

} // namespace monolish
