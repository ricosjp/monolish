#include "../../monolish_internal.hpp"
#include "../monolish_lapack_float.hpp"
#ifndef MONOLISH_USE_MKL
#include "../lapack.h"
#endif

#include <vector>

namespace monolish {

// float
int internal::lapack::sygvd(matrix::Dense<float> &A, matrix::Dense<float> &B,
                            vector<float> &W, const int itype, const char *jobz,
                            const char *uplo) {
#ifdef MONOLISH_USE_GPU
  throw std::logic_error("not yet implemented for GPU");
#else // MONOLISH_USE_GPU
  int info = 0;
  int size = static_cast<int>(A.get_row());
  int lwork = -1;
  std::vector<float> work(1);
  int liwork = -1;
  std::vector<int> iwork(1);
  // workspace query; no error returned
  ssygvd_(&itype, jobz, uplo, &size, A.val.data(), &size, B.val.data(), &size,
          W.data(), work.data(), &lwork, iwork.data(), &liwork, &info);

  // workspace preparation
  lwork = work[0];
  work.resize(lwork);
  liwork = iwork[0];
  iwork.resize(liwork);
  // actual calculation
  ssygvd_(&itype, jobz, uplo, &size, A.val.data(), &size, B.val.data(), &size,
          W.data(), work.data(), &lwork, iwork.data(), &liwork, &info);
  return info;
#endif
}

} // namespace monolish
