#include "../../../include/monolish/common/monolish_dense.hpp"
#include "../../../include/monolish/common/monolish_logger.hpp"
#include "../../../include/monolish/common/monolish_matrix.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {
namespace matrix {

template <typename T> void CRS<T>::convert(COO<T> &coo) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  rowN = coo.get_row();
  colN = coo.get_col();
  val_create_flag = true;

  resize(coo.get_nnz());
  for (size_t i = 0; i < coo.get_nnz(); ++i) {
    data()[i] = coo.data()[i];
  }
  col_ind = coo.col_index;

  // todo not inplace now
  row_ptr.resize(get_row() + 1, 0.0);

  row_ptr[0] = 0;
  size_t c_row = 0;
  for (auto i = decltype(coo.get_nnz()){0}; i < coo.get_nnz(); i++) {

    if ((int)c_row == coo.row_index[i]) {
      row_ptr[c_row + 1] = i + 1;
    } else {
      // c_row = c_row + 1;
      for (auto j = c_row + 1; j < coo.row_index[i]; j++) {
        row_ptr[j + 1] = i;
      }
      c_row = coo.row_index[i];
      row_ptr[c_row + 1] = i + 1;
    }
  }
  compute_hash();
  logger.util_out();
}
template void CRS<double>::convert(COO<double> &coo);
template void CRS<float>::convert(COO<float> &coo);

template <typename T> void CRS<T>::convert(CRS<T> &crs) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  resize(crs.get_nnz());
  col_ind.resize(crs.get_nnz());
  row_ptr.resize(crs.get_row() + 1);
  val_create_flag = true;

  rowN = crs.get_row();
  colN = crs.get_col();
  structure_hash = crs.get_hash();

  if (crs.get_device_mem_stat() == true) {
    throw std::runtime_error(
        "error can not convert CRS->CRS when gpu_status == true");
  } else {
    internal::vcopy(crs.row_ptr.size(), crs.row_ptr.data(), row_ptr.data(),
                    false);
    internal::vcopy(crs.col_ind.size(), crs.col_ind.data(), col_ind.data(),
                    false);
    internal::vcopy(crs.get_nnz(), crs.data(), data(), false);
  }

  logger.util_out();
}
template void CRS<double>::convert(CRS<double> &coo);
template void CRS<float>::convert(CRS<float> &coo);

} // namespace matrix
} // namespace monolish
