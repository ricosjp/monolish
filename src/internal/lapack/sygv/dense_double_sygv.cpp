#include "../../monolish_internal.hpp"
#include "../monolish_lapack_double.hpp"
#ifndef MONOLISH_USE_MKL
#include "../lapack.h"
#endif

#include <vector>

namespace monolish {

// double
bool internal::lapack::sygv(const int itype, const char *jobz, const char *uplo,
                            matrix::Dense<double> &A, matrix::Dense<double> &B,
                            vector<double> &W) {
#ifdef MONOLISH_USE_GPU
  throw std::logic_error("not yet implemented for GPU");
#else // MONOLISH_USE_GPU
  int info = 0;
  int lwork = static_cast<int>((64 + 2) * A.get_row());
  std::vector<double> work(lwork);
  int size = static_cast<int>(A.get_row());
  dsygv_(&itype, jobz, uplo, &size, A.val.data(), &size, B.val.data(), &size,
         W.data(), work.data(), &lwork, &info);
  return (info == 0);
#endif
}

} // namespace monolish