#include "../../../../include/monolish_blas.hpp"
#include "../../../internal/monolish_internal.hpp"
#include "../../matrix/matmul/dense-dense_matmul.hpp"
#include "../mattens/crs-tensor_dense_mattens.hpp"

namespace monolish {
namespace {
// double ///////////////////
template <typename TENS2, typename TENS3>
void tensor_CRS_tensor_Dense_Dtensmul_core(const double &a, const tensor::tensor_CRS<double> &A,
                                             const TENS2 &B, const double &b,
                                             TENS3 &C){
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  auto Ashape = A.get_shape();
  auto Bshape = B.get_shape();
  auto Cshape = C.get_shape();
  int row = Ashape[Ashape.size() - 2];
  int col = Ashape[Ashape.size() - 1];

  assert(col == Bshape[0]);
  std::vector<size_t> ABshape;
  for(size_t i=0; i+1<Ashape.size(); ++i){
    ABshape.push_back(Ashape[i]);
  }
  for(size_t i=1; i<Bshape.size(); ++i){
    ABshape.push_back(Bshape[i]);
  }
  assert(ABshape == Cshape);

  std::vector<size_t> ABshape_tmp = Bshape;
  ABshape_tmp[0] = row;
  size_t ABshape_dim = 1;
  for(size_t i=0; i<ABshape_tmp.size(); ++i){
    ABshape_dim *= ABshape_tmp[i];
  }

  size_t nsum = 0;

  for(size_t d=0; d<A.row_ptrs.size(); ++d){
    matrix::CRS<double> Amat(row, col, A.row_ptrs[d], A.col_inds[d], A.get_val());
    Amat.set_first(A.get_offset() + nsum);
    nsum += A.col_inds[d].size();
    tensor::tensor_Dense<double> Cmat(ABshape_tmp, C.get_val());
    Cmat.set_first(C.get_offset() + d * ABshape_dim);
    CRS_tensor_Dense_Dmattens_core(a, Amat, B, b, Cmat);
  }

  logger.func_out();

}

// float ///////////////////
template <typename TENS2, typename TENS3>
void tensor_CRS_tensor_Dense_Stensmul_core(const float &a, const tensor::tensor_CRS<float> &A,
                                             const TENS2 &B, const float &b,
                                             TENS3 &C){
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  auto Ashape = A.get_shape();
  auto Bshape = B.get_shape();
  auto Cshape = C.get_shape();
  int row = Ashape[Ashape.size() - 2];
  int col = Ashape[Ashape.size() - 1];

  assert(col == Bshape[0]);
  std::vector<size_t> ABshape;
  for(size_t i=0; i+1<Ashape.size(); ++i){
    ABshape.push_back(Ashape[i]);
  }
  for(size_t i=1; i<Bshape.size(); ++i){
    ABshape.push_back(Bshape[i]);
  }
  assert(ABshape == Cshape);

  std::vector<size_t> ABshape_tmp = Bshape;
  ABshape_tmp[0] = row;
  size_t ABshape_dim = 1;
  for(size_t i=0; i<ABshape_tmp.size(); ++i){
    ABshape_dim *= ABshape_tmp[i];
  }

  size_t nsum = 0;

  for(size_t d=0; d<A.row_ptrs.size(); ++d){
    matrix::CRS<float> Amat(row, col, A.row_ptrs[d], A.col_inds[d], A.get_val());
    Amat.set_first(A.get_offset() + nsum);
    nsum += A.col_inds[d].size();
    tensor::tensor_Dense<float> Cmat(ABshape_tmp, C.get_val());
    Cmat.set_first(C.get_offset() + d * ABshape_dim);
    CRS_tensor_Dense_Smattens_core(a, Amat, B, b, Cmat);
  }

  logger.func_out();
}
} // namespace
} // namespace monolish
