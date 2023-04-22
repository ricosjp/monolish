#pragma once
#include "../../matrix/matvec/dense_matvec.hpp"

namespace monolish {
namespace {
std::string get_tensvec_name(std::string func, bool flag) {
  if (flag == true) {
    return func + "_T";
  } else {
    return func + "_N";
  }
}
// double ///////////////////
template <typename VEC1>
void Dtensvec_core(const tensor::tensor_Dense<double> &A, const VEC1 &x,
                   tensor::tensor_Dense<double> &y, bool transA) {
  Logger &logger = Logger::get_instance();
  logger.func_in(get_matvec_name(monolish_func, transA));

  auto Ashape = A.get_shape();
  matrix::Dense<double> Amat;
  vector<double> yvec;
  if (transA) {
    Amat.move(A, Ashape[0], -1);
  } else {
    Amat.move(A, -1, Ashape[Ashape.size() - 1]);
  }
  yvec.move(y, -1);
  Dmatvec_core(Amat, x, yvec, transA);

  logger.func_out();
}

// float ///////////////////
template <typename VEC1>
void Stensvec_core(const tensor::tensor_Dense<float> &A, const VEC1 &x,
                   tensor::tensor_Dense<float> &y, bool transA) {
  Logger &logger = Logger::get_instance();
  logger.func_in(get_matvec_name(monolish_func, transA));

  auto Ashape = A.get_shape();
  matrix::Dense<float> Amat;
  vector<float> yvec;
  if (transA) {
    Amat.move(A, Ashape[0], -1);
  } else {
    Amat.move(A, -1, Ashape[Ashape.size() - 1]);
  }
  yvec.move(y, -1);
  Smatvec_core(Amat, x, yvec, transA);

  logger.func_out();
}
} // namespace

} // namespace monolish
