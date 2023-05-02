#include "../../../../include/monolish_blas.hpp"
#include "../../../internal/monolish_internal.hpp"
#include "../../matrix/matmul/dense-dense_matmul.hpp"

namespace monolish {
namespace {
// double ///////////////////
void Dense_tensor_Dense_Dmattens_core(const double &a,
                                      const matrix::Dense<double> &A,
                                      const tensor::tensor_Dense<double> &B,
                                      const double &b,
                                      tensor::tensor_Dense<double> &C,
                                      bool transA, bool transB) {
  Logger &logger = Logger::get_instance();
  logger.func_in(get_matmul_name(monolish_func, transA, transB));

  auto Bshape = B.get_shape();
  auto Cshape = C.get_shape();
  matrix::Dense<double> Bmat, Cmat;
  if (transB) {
    Bmat.move(B, -1, Bshape[Bshape.size() - 1]);
  } else {
    Bmat.move(B, Bshape[0], -1);
  }
  Cmat.move(C, A.get_row(), Bmat.get_col());
  Dense_Dense_Dmatmul_core(a, A, Bmat, b, Cmat, transA, transB);

  logger.func_out();
}

// float ///////////////////
void Dense_tensor_Dense_Smattens_core(const float &a,
                                      const matrix::Dense<float> &A,
                                      const tensor::tensor_Dense<float> &B,
                                      const float &b,
                                      tensor::tensor_Dense<float> &C,
                                      bool transA, bool transB) {
  Logger &logger = Logger::get_instance();
  logger.func_in(get_matmul_name(monolish_func, transA, transB));

  auto Bshape = B.get_shape();
  auto Cshape = C.get_shape();
  matrix::Dense<float> Bmat, Cmat;
  if (transB) {
    Bmat.move(B, -1, Bshape[Bshape.size() - 1]);
  } else {
    Bmat.move(B, Bshape[0], -1);
  }
  Cmat.move(C, A.get_row(), Bmat.get_col());
  Dense_Dense_Smatmul_core(a, A, Bmat, b, Cmat, transA, transB);

  logger.func_out();
}

} // namespace
} // namespace monolish
