#include "../../../../include/monolish_blas.hpp"
#include "../../../internal/monolish_internal.hpp"
#include "../../matrix/matmul/dense-dense_matmul.hpp"

namespace monolish {
namespace {
// double ///////////////////
template <typename TENS1, typename MAT, typename TENS2>
void tensor_Dense_Dense_Dtensmat_core(const double &a, const TENS1 &A,
                                      const MAT &B, const double &b, TENS2 &C,
                                      bool transA, bool transB) {
  Logger &logger = Logger::get_instance();
  logger.func_in(get_matmul_name(monolish_func, transA, transB));

  auto Ashape = A.get_shape();
  auto Cshape = C.get_shape();
  matrix::Dense<double> Amat, Cmat;
  if (transA) {
    Amat.move(A, Ashape[0], -1);
  } else {
    Amat.move(A, -1, Ashape[Ashape.size() - 1]);
  }
  Cmat.move(C, Amat.get_row(), B.get_col());
  Dense_Dense_Dmatmul_core(a, Amat, B, b, Cmat, transA, transB);

  logger.func_out();
}

// float ///////////////////
template <typename TENS1, typename MAT, typename TENS2>
void tensor_Dense_Dense_Stensmat_core(const float &a, const TENS1 &A,
                                      const MAT &B, const float &b, TENS2 &C,
                                      bool transA, bool transB) {
  Logger &logger = Logger::get_instance();
  logger.func_in(get_matmul_name(monolish_func, transA, transB));

  auto Ashape = A.get_shape();
  auto Cshape = C.get_shape();
  matrix::Dense<float> Amat, Cmat;
  if (transA) {
    Amat.move(A, Ashape[0], -1);
  } else {
    Amat.move(A, -1, Ashape[Ashape.size() - 1]);
  }
  Cmat.move(C, Amat.get_row(), B.get_col());
  Dense_Dense_Smatmul_core(a, Amat, B, b, Cmat, transA, transB);

  logger.func_out();
}
} // namespace
} // namespace monolish
