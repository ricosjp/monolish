#include "../../../include/monolish/common/monolish_dense.hpp"
#include "../../../include/monolish/common/monolish_logger.hpp"
#include "../../../include/monolish/common/monolish_matrix.hpp"
#include "../../../include/monolish/common/monolish_tensor.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {
namespace tensor {

template <typename T>
void tensor_COO<T>::convert(const tensor::tensor_CRS<T> &crs) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  set_shape(crs.get_shape());
  val_create_flag = true;
  resize(crs.get_nnz());
  index.resize(crs.get_nnz());

  std::vector<size_t> shape = crs.get_shape();
  int dim = shape.size();
  std::vector<size_t> pos(dim);
  size_t nz = 0;

  for (auto d = 0; d < crs.col_inds.size(); ++d) {
    int d_copy = d;
    for (int k = dim - 3; k >= 0; --k) {
      pos[k] = d_copy % shape[k];
      d_copy /= shape[k];
    }
    for (auto i = 0; i < crs.row_ptrs[d].size(); ++i) {
      for (size_t j = crs.row_ptrs[d][i]; j < crs.row_ptrs[d][i + 1]; ++j) {
        pos[dim - 2] = i;
        pos[dim - 1] = crs.col_inds[d][j];
        index.push_back(pos);
        data()[nz] = crs.begin()[nz];
        nz++;
      }
    }
  }

  logger.util_out();
}

template void
tensor_COO<double>::convert(const tensor::tensor_CRS<double> &crs);
template void tensor_COO<float>::convert(const tensor::tensor_CRS<float> &crs);

template <typename T>
void tensor_COO<T>::convert(const tensor::tensor_Dense<T> &dense) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  set_shape(dense.get_shape());
  val_create_flag = true;
  resize(dense.get_nnz());
  index.clear();
  size_t nz = 0;

  std::vector<size_t> pos(shape.size());
  for (auto i = decltype(dense.get_nnz()){0}; i < dense.get_nnz(); i++) {
    if (dense.data()[i] != 0) {
      auto i_copy = i;
      for (int j = shape.size() - 1; j >= 0; --j) {
        pos[j] = i_copy % shape[j];
        i_copy /= shape[j];
      }
      index.push_back(pos);
      data()[nz] = dense.begin()[i];
      nz++;
    }
  }
  resize(nz);

  logger.util_out();
}
template void
tensor_COO<double>::convert(const tensor::tensor_Dense<double> &dense);
template void
tensor_COO<float>::convert(const tensor::tensor_Dense<float> &dense);

} // namespace tensor
} // namespace monolish
