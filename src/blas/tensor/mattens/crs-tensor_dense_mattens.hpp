#include "../../../../include/monolish_blas.hpp"
#include "../../../internal/monolish_internal.hpp"
#include "../../matrix/matmul/crs-dense_matmul.hpp"

namespace monolish {
namespace {
// double ///////////////////
template <typename TENS1, typename TENS2>
void CRS_tensor_Dense_Dmattens_core(const double &a,
                                    const matrix::CRS<double> &A,
                                    const TENS1 &B, const double &b, TENS2 &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  auto Bshape = B.get_shape();
  auto Cshape = C.get_shape();
  matrix::Dense<double> Bmat, Cmat;
  Bmat.move(B, Bshape[0], -1);
  Cmat.move(C, A.get_row(), Bmat.get_col());
  blas::matmul(a, A, Bmat, b, Cmat);

  logger.func_out();
}

// float ///////////////////
template <typename TENS1, typename TENS2>
void CRS_tensor_Dense_Smattens_core(const float &a, const matrix::CRS<float> &A,
                                    const TENS1 &B, const float &b, TENS2 &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  auto Bshape = B.get_shape();
  auto Cshape = C.get_shape();
  matrix::Dense<float> Bmat, Cmat;
  Bmat.move(B, Bshape[0], -1);
  Cmat.move(C, A.get_row(), Bmat.get_col());
  blas::matmul(a, A, Bmat, b, Cmat);

  logger.func_out();
}

} // namespace
} // namespace monolish
