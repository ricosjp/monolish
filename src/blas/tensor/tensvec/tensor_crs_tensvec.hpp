#pragma once
#include "../../matrix/matvec/dense_matvec.hpp"

namespace monolish {
namespace {
// double ///////////////////
template <typename VEC1, typename TENS2>
void Dtensvec_core(const double &a, const tensor::tensor_CRS<double> &A,
                   const VEC1 &x, const double &b, TENS2 &C, bool transA) {
  Logger &logger = Logger::get_instance();
  logger.func_in(get_matvec_name(monolish_func, transA));

  auto Ashape = A.get_shape();
  auto Cshape = C.get_shape();
  int row = Ashape[Ashape.size() - 2];
  int col = Ashape[Ashape.size() - 1];
  assert(Ashape[Ashape.size() - 1] == x.size());
  Cshape.push_back(col);
  assert(Ashape.size() == Cshape.size());
  assert(Ashape == Cshape);
  assert(util::is_same_device_mem_stat(A, x, C));

  int nsum = 0;
  for (size_t d = 0; d < A.row_ptrs.size(); ++d) {
    std::shared_ptr<double> tmp(new double[A.col_inds[d].size()],
                                std::default_delete<double[]>());
    for (size_t i = 0; i < A.col_inds[d].size(); ++i) {
      tmp.get()[i] = A.begin()[i + nsum];
    }
    matrix::CRS<double> Amat(row, col, A.row_ptrs[d], A.col_inds[d], tmp);
    vector<double> Cvec(row);
    for (size_t i = 0; i < row; ++i) {
      Cvec.begin()[i] = C.begin()[d * row + i];
    }
    Dmatvec_core(a, Amat, x, b, Cvec, transA);
    for (size_t i = 0; i < row; ++i) {
      C.begin()[d * row + i] = Cvec.begin()[i];
    }
    nsum += A.col_inds[d].size();
  }

  logger.func_out();
}

// float ///////////////////
template <typename VEC1, typename TENS2>
void Stensvec_core(const float &a, const tensor::tensor_CRS<float> &A,
                   const VEC1 &x, const float &b, TENS2 &C, bool transA) {
  Logger &logger = Logger::get_instance();
  logger.func_in(get_matvec_name(monolish_func, transA));

  auto Ashape = A.get_shape();
  auto Cshape = C.get_shape();
  int row = Ashape[Ashape.size() - 2];
  int col = Ashape[Ashape.size() - 1];
  assert(Ashape[Ashape.size() - 1] == x.size());
  Cshape.push_back(col);
  assert(Ashape.size() == Cshape.size());
  assert(Ashape == Cshape);
  assert(util::is_same_device_mem_stat(A, x, C));

  int nsum = 0;
  for (size_t d = 0; d < A.row_ptrs.size(); ++d) {
    std::shared_ptr<float> tmp(new float[A.col_inds[d].size()],
                               std::default_delete<float[]>());
    for (size_t i = 0; i < A.col_inds[d].size(); ++i) {
      tmp.get()[i] = A.begin()[i + nsum];
    }
    matrix::CRS<float> Amat(row, col, A.row_ptrs[d], A.col_inds[d], tmp);
    vector<float> Cvec(row);
    for (size_t i = 0; i < row; ++i) {
      Cvec.begin()[i] = C.begin()[d * row + i];
    }
    Smatvec_core(a, Amat, x, b, Cvec, transA);
    for (size_t i = 0; i < row; ++i) {
      C.begin()[d * row + i] = Cvec.begin()[i];
    }
    nsum += A.col_inds[d].size();
  }

  logger.func_out();
}
} // namespace

} // namespace monolish
