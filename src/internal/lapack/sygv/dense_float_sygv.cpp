#include "../../monolish_internal.hpp"
#include "../monolish_lapack_float.hpp"

#include <vector>

#ifdef MONOLISH_USE_MKL
#define ssygv_ ssygv
#else
#include <lapacke.h>
#endif

namespace monolish {

// float
bool internal::lapack::sygv(const int itype, const char *jobz, const char *uplo,
                            matrix::Dense<float> &A, matrix::Dense<float> &B,
                            vector<float> &W) {
#ifdef MONOLISH_USE_GPU
  throw std::logic_error("not yet implemented for GPU");
#else // MONOLISH_USE_GPU
  int info = 0;
  int lwork = static_cast<int>((64 + 2) * A.get_row());
  std::vector<float> work(lwork);
  int size = static_cast<int>(A.get_row());
#ifdef MONOLISH_USE_MKL
  ssygv_(&itype, jobz, uplo, &size, A.val.data(), &size, B.val.data(), &size,
         W.data(), work.data(), &lwork, &info);
#else
  info = LAPACKE_ssygv_work(LAPACK_COL_MAJOR, itype, jobz[0], uplo[0], size,
                            A.val.data(), size, B.val.data(), size, W.data(),
                            work.data(), lwork);
#endif
  return (info == 0);
#endif
}

} // namespace monolish
