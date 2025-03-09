#pragma once
#include "../../../internal/monolish_internal.hpp"
#include "../../matrix/matvec/crs_matvec.hpp"
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
  assert(col == x.size());
  Cshape.push_back(col);
  assert(Ashape.size() == Cshape.size());
  assert(Ashape == Cshape);
  assert(util::is_same_device_mem_stat(A, x, C));

  int nsum = 0;
  for (size_t d = 0; d < A.row_ptrs.size(); ++d) {
    std::vector<double> Aval(A.col_inds[d].size());
    matrix::CRS<double> Amat(row, col, A.row_ptrs[d], A.col_inds[d], Aval);
    std::vector<double> Cval(row);
    monolish::vector<double> Cvec(Cval);
    if (A.get_device_mem_stat()) {
      Amat.send();
      Cvec.send();
    }
    internal::vcopy(Aval.size(), A.begin() + nsum, Amat.begin(),
                    A.get_device_mem_stat());
    internal::vcopy(Cval.size(), C.begin() + d * row, Cvec.begin(),
                    A.get_device_mem_stat());
    Dmatvec_core(a, Amat, x, b, Cvec, transA);
    nsum += A.col_inds[d].size();
    internal::vcopy(Cval.size(), Cvec.begin(), C.begin() + d * row,
                    A.get_device_mem_stat());
    if (A.get_device_mem_stat()) {
      Amat.recv();
      Cvec.recv();
    }
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
  assert(col == x.size());
  Cshape.push_back(col);
  assert(Ashape.size() == Cshape.size());
  assert(Ashape == Cshape);
  assert(util::is_same_device_mem_stat(A, x, C));

  int nsum = 0;
  for (size_t d = 0; d < A.row_ptrs.size(); ++d) {
    std::vector<float> Aval(A.col_inds[d].size());
    matrix::CRS<float> Amat(row, col, A.row_ptrs[d], A.col_inds[d], Aval);
    std::vector<float> Cval(row);
    monolish::vector<float> Cvec(Cval);
    if (A.get_device_mem_stat()) {
      Amat.send();
      Cvec.send();
    }
    internal::vcopy(Aval.size(), A.begin() + nsum, Amat.begin(),
                    A.get_device_mem_stat());
    internal::vcopy(Cval.size(), C.begin() + d * row, Cvec.begin(),
                    A.get_device_mem_stat());
    Smatvec_core(a, Amat, x, b, Cvec, transA);
    nsum += A.col_inds[d].size();
    internal::vcopy(Cval.size(), Cvec.begin(), C.begin() + d * row,
                    A.get_device_mem_stat());
    if (A.get_device_mem_stat()) {
      Amat.recv();
      Cvec.recv();
    }
  }

  logger.func_out();
}
} // namespace

} // namespace monolish
