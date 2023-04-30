#include "../../../../include/monolish_blas.hpp"
#include "../../../internal/monolish_internal.hpp"
#include "../../matrix/matmul/crs-dense_matmul.hpp"

namespace monolish {
namespace {
// double ///////////////////
void CRS_tensor_Dense_Dmattens_core(const double &a,
                                    const matrix::CRS<double> &A,
                                    const tensor::tensor_Dense<double> &B,
                                    const double &b,
                                    tensor::tensor_Dense<double> &C) {
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
void CRS_tensor_Dense_Smattens_core(const float &a, const matrix::CRS<float> &A,
                                    const tensor::tensor_Dense<float> &B,
                                    const float &b,
                                    tensor::tensor_Dense<float> &C) {
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
