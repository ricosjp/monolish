#include "../../../../include/monolish_lapack.hpp"
#include "../../../monolish_internal.hpp"

#include <vector>

namespace monolish {

// double
bool lapack::syev(const char *jobz, const char *uplo, matrix::Dense<double> &A,
                  vector<double> &W) {
#if MONOLISH_USE_GPU
  throw std::logic_eror("not yet implemented for GPU");
#else // MONOLISH_USE_GPU
  int info = 0;
  std::size_t lwork = (64+2)*A.get_row();
  std::vector<double> work(lwork);
  dsyev(jobz, uplo, A.get_row(), A.data(), A.get_row(), W.data(), work.data(), &lwork, &info);
  return (info==0);
#endif
}

} // namespace monolish    
