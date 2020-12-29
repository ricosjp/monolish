#include "../../../include/monolish_lapack.hpp"
#include "../../internal/monolish_internal.hpp"

#include <vector>
#include <exception>

namespace monolish {

// double
bool lapack::syev(const char *jobz, const char *uplo, matrix::Dense<double> &A,
                  vector<double> &W) {
#if MONOLISH_USE_GPU
  throw std::logic_eror("not yet implemented for GPU");
#else // MONOLISH_USE_GPU
  int info = 0;
  int lwork = static_cast<int>((64+2)*A.get_row());
  std::vector<double> work(lwork);
  int size = static_cast<int>(A.get_row());
  dsyev(jobz, uplo, &size, A.val.data(), &size, W.data(), work.data(), &lwork, &info);
  return (info==0);
#endif
}

} // namespace monolish    
