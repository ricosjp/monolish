#include "../../../include/monolish_blas.hpp"
#include "../../monolish_internal.hpp"

#ifdef USE_GPU
#include <cublas_v2.h>
#else
#include <mkl_cblas.h>
#endif

#include <mkl.h>

namespace monolish {

void mkl_tanh(size_t N, double* vec){
    vdTanh(N, vec, vec);
}

void mkl_tanh(size_t N, float* vec){
    vsTanh(N, vec, vec);
}

template <typename T> void vector<T>::tanh() {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  vector<T> ans(val.size());

  T *vald = val.data();
  T *ansd = ans.data();
  size_t size = val.size();

#if USE_GPU
#pragma acc data present(vald [0:size], ansd [0:size])
#pragma acc parallel
#pragma acc loop independent
  for (size_t i = 0; i < size; i++) {
     vald[i] = tanh(vald[i]);
  }
#else
//#if USE_MKL
  for (size_t i = 0; i < size; i++) {
   mkl_tanh(size, vald);
  }
// #else
// #pragma omp parallel for
//   for (size_t i = 0; i < size; i++) {
//    vald[i] = std::tanh(vald[i]);
//   }
// #endif
#endif

  logger.func_out();
}

template void vector<double>::tanh();
template void vector<float>::tanh();

} // namespace monolish
