#include "../../../../include/monolish_blas.hpp"
#include "../../../internal/monolish_internal.hpp"
#include "../../matrix/matmul/dense-dense_matmul.hpp"

namespace monolish {
namespace {
// double ///////////////////
template <typename MAT, typename TENS1, typename TENS2>
void Dense_tensor_Dense_Dmattens_core(const double &a, const MAT &A,
                                      const TENS1 &B, const double &b, TENS2 &C,
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
template <typename MAT, typename TENS1, typename TENS2>
void Dense_tensor_Dense_Smattens_core(const float &a, const MAT &A,
                                      const TENS1 &B, const float &b, TENS2 &C,
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
