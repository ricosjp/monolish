#include "../../../../include/monolish_blas.hpp"
#include "../../../internal/monolish_internal.hpp"
#include "../../matrix/matmul/crs-dense_matmul.hpp"
#include "../../matrix/matmul/dense-dense_matmul.hpp"

namespace monolish {
namespace {
// double ///////////////////
template <typename MAT, typename TENS2>
void tensor_CRS_Dense_Dtensmat_core(const double &a,
                                    const tensor::tensor_CRS<double> &A,
                                    const MAT &B, const double &b, TENS2 &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  auto Ashape = A.get_shape();
  auto Cshape = C.get_shape();
  int row = Ashape[Ashape.size() - 2];
  int col = Ashape[Ashape.size() - 1];
  assert(Ashape[Ashape.size() - 1] == B.get_row());
  assert(B.get_col() == Cshape[Cshape.size() - 1]);
  assert(Ashape.size() == Cshape.size());
  Ashape[Ashape.size() - 1] = Cshape[Cshape.size() - 1];
  assert(Ashape == Cshape);
  Ashape[Ashape.size() - 1] = col;
  assert(util::is_same_device_mem_stat(A, B, C));

  size_t nsum = 0;

  for (size_t d = 0; d < A.row_ptrs.size(); ++d) {
    std::vector<double> Aval(A.col_inds[d].size());
    matrix::CRS<double> Amat(row, col, A.row_ptrs[d], A.col_inds[d], Aval);
    std::vector<double> Cval(row * B.get_col());
    matrix::Dense<double> Cmat(row, B.get_col(), Cval);
    if (A.get_device_mem_stat()) {
      Amat.send();
      Cmat.send();
    }
    internal::vcopy(Aval.size(), A.begin() + nsum, Amat.begin(),
                    A.get_device_mem_stat());
    internal::vcopy(Cval.size(), C.begin() + d * row * B.get_col(),
                    Cmat.begin(), A.get_device_mem_stat());
    CRS_Dense_Dmatmul_core(a, Amat, B, b, Cmat);
    internal::vcopy(Cval.size(), Cmat.begin(),
                    C.begin() + d * row * B.get_col(), A.get_device_mem_stat());
    if (A.get_device_mem_stat()) {
      Amat.recv();
      Cmat.recv();
    }
    nsum += A.col_inds[d].size();
  }

  logger.func_out();
}

// float ///////////////////
template <typename MAT, typename TENS2>
void tensor_CRS_Dense_Stensmat_core(const float &a,
                                    const tensor::tensor_CRS<float> &A,
                                    const MAT &B, const float &b, TENS2 &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  auto Ashape = A.get_shape();
  auto Cshape = C.get_shape();
  int row = Ashape[Ashape.size() - 2];
  int col = Ashape[Ashape.size() - 1];
  assert(Ashape[Ashape.size() - 1] == B.get_row());
  assert(B.get_col() == Cshape[Cshape.size() - 1]);
  assert(Ashape.size() == Cshape.size());
  Ashape[Ashape.size() - 1] = Cshape[Cshape.size() - 1];
  assert(Ashape == Cshape);
  Ashape[Ashape.size() - 1] = col;
  assert(util::is_same_device_mem_stat(A, B, C));

  size_t nsum = 0;

  for (size_t d = 0; d < A.row_ptrs.size(); ++d) {
    std::vector<float> Aval(A.col_inds[d].size());
    matrix::CRS<float> Amat(row, col, A.row_ptrs[d], A.col_inds[d], Aval);
    std::vector<float> Cval(row * B.get_col());
    matrix::Dense<float> Cmat(row, B.get_col(), Cval);
    if (A.get_device_mem_stat()) {
      Amat.send();
      Cmat.send();
    }
    internal::vcopy(Aval.size(), A.begin() + nsum, Amat.begin(),
                    A.get_device_mem_stat());
    internal::vcopy(Cval.size(), C.begin() + d * row * B.get_col(),
                    Cmat.begin(), A.get_device_mem_stat());
    CRS_Dense_Smatmul_core(a, Amat, B, b, Cmat);
    internal::vcopy(Cval.size(), Cmat.begin(),
                    C.begin() + d * row * B.get_col(), A.get_device_mem_stat());
    if (A.get_device_mem_stat()) {
      Amat.recv();
      Cmat.recv();
    }
    nsum += A.col_inds[d].size();
  }

  logger.func_out();
}
} // namespace
} // namespace monolish
