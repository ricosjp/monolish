#include "../../../include/monolish/common/monolish_dense.hpp"
#include "../../../include/monolish/common/monolish_logger.hpp"
#include "../../../include/monolish/common/monolish_matrix.hpp"
#include "../../../include/monolish/common/monolish_tensor.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {
namespace tensor {

template <typename T>
void tensor_CRS<T>::convert(const tensor::tensor_COO<T> &coo) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  set_shape(coo.get_shape());
  val_create_flag = true;
  resize(coo.get_nnz());
  int dim = coo.get_shape().size();
  size_t upper_d = 1;
  for (int i = 0; i < dim - 2; ++i) {
    upper_d *= coo.get_shape()[i];
  }
  row_ptrs.resize(upper_d);
  col_inds.resize(upper_d);
  for (size_t i = 0; i < coo.get_nnz(); ++i) {
    begin()[i] = coo.begin()[i];
  }

  for (size_t ud = 0; ud < upper_d; ++ud) {
    row_ptrs[ud].resize(shape[dim - 2] + 1);
  }

  std::vector<size_t> c_rows(upper_d);
  int new_i = 0;
  int prev_upper_d_tmp = 0;
  for (auto i = 0; i < coo.get_nnz(); ++i) {
    int upper_d_tmp = 0;
    for (auto d = 0; d < dim - 2; ++d) {
      upper_d_tmp *= shape[d];
      upper_d_tmp += coo.index[i][d];
    }

    if (upper_d_tmp != prev_upper_d_tmp) {
      new_i = 0;
      prev_upper_d_tmp = upper_d_tmp;
    }

    if ((int)c_rows[upper_d_tmp] == coo.index[i][dim - 2]) {
      row_ptrs[upper_d_tmp][c_rows[upper_d_tmp] + 1] = new_i + 1;
    } else {
      c_rows[upper_d_tmp] = c_rows[upper_d_tmp] + 1;
      row_ptrs[upper_d_tmp][c_rows[upper_d_tmp] + 1] = new_i + 1;
    }
    col_inds[upper_d_tmp].push_back(coo.index[i][dim - 1]);

    new_i++;
  }

  compute_hash();
  logger.util_out();
}
template void
tensor_CRS<double>::convert(const tensor::tensor_COO<double> &coo);
template void tensor_CRS<float>::convert(const tensor::tensor_COO<float> &coo);

template <typename T>
void tensor_CRS<T>::convert(const tensor::tensor_CRS<T> &tensor_crs) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  resize(tensor_crs.get_nnz());
  set_shape(tensor_crs.get_shape());
  col_inds.resize(tensor_crs.col_inds.size());
  row_ptrs.resize(tensor_crs.row_ptrs.size());
  for (std::size_t i = 0; i < tensor_crs.col_inds.size(); ++i) {
    col_inds[i].resize(tensor_crs.col_inds[i].size());
    row_ptrs[i].resize(tensor_crs.row_ptrs[i].size() + 1);
  }
  val_create_flag = true;

  structure_hash = tensor_crs.get_hash();

  if (tensor_crs.get_device_mem_stat() == true) {
    throw std::runtime_error(
        "error can not convert CRS->CRS when gpu_status == true");
  } else {
    for (size_t i = 0; i < tensor_crs.col_inds.size(); ++i) {
      internal::vcopy(tensor_crs.row_ptrs[i].size(),
                      tensor_crs.row_ptrs[i].data(), row_ptrs[i].data(), false);
      internal::vcopy(tensor_crs.col_inds[i].size(),
                      tensor_crs.col_inds[i].data(), col_inds[i].data(), false);
    }
    internal::vcopy(tensor_crs.get_nnz(), tensor_crs.data(), data(), false);
  }

  logger.util_out();
}
template void
tensor_CRS<double>::convert(const tensor::tensor_CRS<double> &tensor_crs);
template void
tensor_CRS<float>::convert(const tensor::tensor_CRS<float> &tensor_crs);

template <typename T> void tensor_CRS<T>::convert(const matrix::CRS<T> &crs) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  resize(crs.get_nnz());
  set_shape({crs.get_row(), crs.get_col()});
  col_inds.resize(1);
  row_ptrs.resize(1);
  col_inds[0].resize(crs.get_nnz());
  row_ptrs[0].resize(crs.get_row() + 1);
  val_create_flag = true;

  structure_hash = crs.get_hash();

  if (crs.get_device_mem_stat() == true) {
    throw std::runtime_error(
        "error can not convert CRS->CRS when gpu_status == true");
  } else {
    internal::vcopy(crs.row_ptr.size(), crs.row_ptr.data(), row_ptrs[0].data(),
                    false);
    internal::vcopy(crs.col_ind.size(), crs.col_ind.data(), col_inds[0].data(),
                    false);
    internal::vcopy(crs.get_nnz(), crs.data(), data(), false);
  }

  logger.util_out();
}
template void tensor_CRS<double>::convert(const matrix::CRS<double> &crs);
template void tensor_CRS<float>::convert(const matrix::CRS<float> &crs);

} // namespace tensor
} // namespace monolish
