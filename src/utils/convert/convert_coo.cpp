#include "../../../include/common/monolish_dense.hpp"
#include "../../../include/common/monolish_logger.hpp"
#include "../../../include/common/monolish_matrix.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {
namespace matrix {

template <typename T> void COO<T>::convert(const CRS<T> &crs) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  set_row(crs.get_row());
  set_col(crs.get_col());
  set_nnz(crs.get_nnz());

  row_index.resize(nnz);
  col_index.resize(nnz);
  val.resize(nnz);

  for (size_t i = 0; i < get_row(); i++) {
    for (size_t j = (size_t)crs.row_ptr[i]; j < (size_t)crs.row_ptr[i + 1];
         j++) {
      row_index[j] = i;
      col_index[j] = crs.col_ind[j];
      val[j] = crs.val[j];
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
  row_index.resize(0);
  col_index.resize(0);
  val.resize(0);
  size_t nz = 0;

  for (size_t i = 0; i < dense.get_row(); i++) {
    for (size_t j = 0; j < dense.get_col(); j++) {
      if (dense.at(i, j) != 0) {
        row_index.push_back(i);
        col_index.push_back(j);
        val.push_back(dense.at(i, j));
        nz++;
      }
    }
  }
  set_nnz(nz);

  logger.util_out();
}
template void COO<double>::convert(const Dense<double> &dense);
template void COO<float>::convert(const Dense<float> &dense);

}
}
