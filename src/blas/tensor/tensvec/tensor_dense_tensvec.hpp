#pragma once
#include "../../matrix/matvec/dense_matvec.hpp"

namespace monolish {
namespace {
// double ///////////////////
template <typename TENS1, typename VEC1, typename TENS2>
void Dtensvec_core(const TENS1 &A, const VEC1 &x, TENS2 &y, bool transA) {
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
template <typename TENS1, typename VEC1, typename TENS2>
void Stensvec_core(const TENS1 &A, const VEC1 &x, TENS2 &y, bool transA) {
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
