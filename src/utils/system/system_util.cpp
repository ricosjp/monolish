#include "../../../include/monolish_blas.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {

bool util::build_with_avx() {
#if MONOLISH_USE_AVX
  return true;
#else
  return false;
#endif
}

bool util::build_with_avx2() {
#if MONOLISH_USE_AVX2
  return true;
#else
  return false;
#endif
}

bool util::build_with_avx512() {
#if MONOLISH_USE_AVX512
  return true;
#else
  return false;
#endif
}

bool util::build_with_mpi() {
#if MONOLISH_USE_MPI
  return true;
#else
  return false;
#endif
}

bool util::build_with_gpu() {
#if MONOLISH_USE_NVIDIA_GPU
  return true;
#else
  return false;
#endif
}

bool util::build_with_mkl() {
#if MONOLISH_USE_MKL
  return true;
#else
  return false;
#endif
}

bool util::build_with_lapack() {
#if MONOLISH_USE_LAPACK
  return true;
#else
  return false;
#endif
}

bool util::build_with_cblas() {
#if MONOLISH_USE_MKL
  return false;
#else
  return true;
#endif
}

} // namespace monolish
