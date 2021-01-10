#include "../../../include/monolish_lapack.hpp"
#include "../../internal/monolish_internal.hpp"

#include <vector>

#ifdef MONOLISH_USE_MKL
#define ssyev_ ssyev
#else
#include <lapacke.h>
#endif

namespace monolish {

// float
bool lapack::syev(const char *jobz, const char *uplo, matrix::Dense<float> &A,
                  vector<float> &W) {
#ifdef MONOLISH_USE_GPU
  throw std::logic_error("not yet implemented for GPU");
#else // MONOLISH_USE_GPU
  int info = 0;
  int lwork = static_cast<int>((64 + 2) * A.get_row());
  std::vector<float> work(lwork);
  int size = static_cast<int>(A.get_row());
#ifdef MONOLISH_USE_MKL
  ssyev_(jobz, uplo, &size, A.val.data(), &size, W.data(), work.data(), &lwork,
         &info);
#else
  info = LAPACKE_ssyev_work(LAPACK_COL_MAJOR, jobz[0], uplo[0], size,
                            A.val.data(), size, W.data(), work.data(), lwork);
#endif
  return (info == 0);
#endif
}

} // namespace monolish
