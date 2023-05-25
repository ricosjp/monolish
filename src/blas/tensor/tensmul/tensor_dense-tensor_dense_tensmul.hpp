#include "../../../../include/monolish_blas.hpp"
#include "../../../internal/monolish_internal.hpp"
#include "../../matrix/matmul/dense-dense_matmul.hpp"

namespace monolish {
namespace {
// double ///////////////////
template <typename TENS1, typename TENS2, typename TENS3>
void tensor_Dense_tensor_Dense_Dtensmul_core(const double &a, const TENS1 &A,
                                             const TENS2 &B, const double &b,
                                             TENS3 &C, bool transA,
                                             bool transB) {
  Logger &logger = Logger::get_instance();
  logger.func_in(get_matmul_name(monolish_func, transA, transB));

  auto Ashape = A.get_shape();
  auto Bshape = B.get_shape();
  auto Cshape = C.get_shape();
  matrix::Dense<double> Amat, Bmat, Cmat;
  Amat.move(A, -1, Ashape[Ashape.size() - 1]);
  Bmat.move(B, Bshape[0], -1);
  Cmat.move(C, Amat.get_row(), Bmat.get_col());
  Dense_Dense_Dmatmul_core(a, Amat, Bmat, b, Cmat, transA, transB);

  logger.func_out();
}

// float ///////////////////
template <typename TENS1, typename TENS2, typename TENS3>
void tensor_Dense_tensor_Dense_Stensmul_core(const float &a, const TENS1 &A,
                                             const TENS2 &B, const float &b,
                                             TENS3 &C, bool transA,
                                             bool transB) {
  Logger &logger = Logger::get_instance();
  logger.func_in(get_matmul_name(monolish_func, transA, transB));

  auto Ashape = A.get_shape();
  auto Bshape = B.get_shape();
  auto Cshape = C.get_shape();
  matrix::Dense<float> Amat, Bmat, Cmat;
  Amat.move(A, -1, Ashape[Ashape.size() - 1]);
  Bmat.move(B, Bshape[0], -1);
  Cmat.move(C, Amat.get_row(), Bmat.get_col());
  Dense_Dense_Smatmul_core(a, Amat, Bmat, b, Cmat, transA, transB);

  logger.func_out();
}
} // namespace
} // namespace monolish
