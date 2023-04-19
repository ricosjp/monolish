#include"../../matrix/matmul/dense-dense_matmul.hpp"
#include "../../../../include/monolish_blas.hpp"
#include "../../../internal/monolish_internal.hpp"

namespace monolish {
namespace {
std::string get_matmul_name(std::string func, bool transA, bool transB) {
  func += "_";
  if (transA == true) {
    func += "T";
  } else {
    func += "N";
  }
  if (transB == true) {
    func += "T";
  } else {
    func += "N";
  }
  return func;
}

// double ///////////////////
void tensor_Dense_Dense_Dmatmul_core(const double &a, const matrix::Dense<double> &A,
                              const tensor::tensor_Dense<double> &B, const double &b,
                              tensor::tensor_Dense<double> &C, bool transA,
                              bool transB) {
  Logger &logger = Logger::get_instance();
  logger.func_in(get_matmul_name(monolish_func, transA, transB));

  auto Ashape = A.get_shape();
  auto Cshape = C.get_shape();
  matrix::Dense<double> Amat, Cmat;
  Amat.move(A, -1, Ashape[Ashape.size()-1]);
  Cmat.move(C, Amat.get_row(), B.get_col());
  Dense_Dense_Dmatmul_core(a, Amat, B, b, Cmat, transA, transB);

  logger.func_out();
}

// float ///////////////////
void tensor_Dense_Dense_Smatmul_core(const float &a, const tensor::tensor_Dense<float> &A,
                              const matrix::Dense<float> &B, const float &b,
                              tensor::tensor_Dense<float> &C, bool transA,
                              bool transB) {
  Logger &logger = Logger::get_instance();
  logger.func_in(get_matmul_name(monolish_func, transA, transB));


  auto Ashape = A.get_shape();
  auto Cshape = C.get_shape();
  matrix::Dense<float> Amat, Cmat;
  Amat.move(A, -1, Ashape[Ashape.size()-1]);
  Cmat.move(C, Amat.get_row(), B.get_col());
  Dense_Dense_Smatmul_core(a, Amat, B, b, Cmat, transA, transB);

  logger.func_out();
}
} // namespace
} // namespace monolish
