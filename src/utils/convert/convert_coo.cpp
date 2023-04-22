#include "../../../include/monolish/common/monolish_dense.hpp"
#include "../../../include/monolish/common/monolish_logger.hpp"
#include "../../../include/monolish/common/monolish_matrix.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {
namespace matrix {

template <typename T> void COO<T>::convert(const CRS<T> &crs) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  set_row(crs.get_row());
  set_col(crs.get_col());
  val_create_flag = true;
  resize(crs.get_nnz());

  row_index.resize(crs.get_nnz());
  col_index.resize(crs.get_nnz());

  for (auto i = decltype(get_row()){0}; i < get_row(); i++) {
    for (auto j = (size_t)crs.row_ptr[i]; j < (size_t)crs.row_ptr[i + 1]; j++) {
      row_index[j] = i;
      col_index[j] = crs.col_ind[j];
      data()[j] = crs.data()[j];
    }
  }

  logger.util_out();
}
template void COO<double>::convert(const CRS<double> &crs);
template void COO<float>::convert(const CRS<float> &crs);

template <typename T> void COO<T>::convert(const Dense<T> &dense) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  set_row(dense.get_row());
  set_col(dense.get_col());
  val_create_flag = true;
  resize(dense.get_row() * dense.get_col());
  row_index.clear();
  col_index.clear();
  size_t nz = 0;

  for (auto i = decltype(dense.get_row()){0}; i < dense.get_row(); i++) {
    for (auto j = decltype(dense.get_col()){0}; j < dense.get_col(); j++) {
      if (dense.at(i, j) != 0) {
        row_index.push_back(i);
        col_index.push_back(j);
        data()[nz] = dense.at(i, j);
        nz++;
      }
    }
  }
  resize(nz);

  logger.util_out();
}
template void COO<double>::convert(const Dense<double> &dense);
template void COO<float>::convert(const Dense<float> &dense);

template <typename T>
void COO<T>::convert(const LinearOperator<T> &linearoperator) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  Dense<T> dense(linearoperator.get_row(), linearoperator.get_col());
  linearoperator.convert_to_Dense(dense);
  convert(dense);

  logger.util_out();
}

template void
COO<double>::convert(const LinearOperator<double> &linearoperator);
template void COO<float>::convert(const LinearOperator<float> &linearoperator);

} // namespace matrix
} // namespace monolish
