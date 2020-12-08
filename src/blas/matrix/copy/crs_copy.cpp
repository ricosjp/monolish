#include "../../../../include/monolish_blas.hpp"
#include "../../../internal/monolish_internal.hpp"

namespace monolish {
namespace matrix {

// copy
template <typename T> CRS<T> CRS<T>::copy() {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  CRS<T> ans(get_row(), get_col(), get_nnz());

  // gpu copy
  if (get_device_mem_stat()) {
    ans.send();
    internal::vcopy((get_row()+1), row_ptr.data(), ans.row_ptr.data(), true);
    internal::vcopy(get_nnz(), col_ind.data(), ans.col_ind.data(), true);
    internal::vcopy(get_nnz(), val.data(), ans.val.data(), true);
  } 

    internal::vcopy((get_row()+1), row_ptr.data(), ans.row_ptr.data(), false);
    internal::vcopy(get_nnz(), col_ind.data(), ans.col_ind.data(), false);
    internal::vcopy(get_nnz(), val.data(), ans.val.data(), false);

  logger.util_out();
  return ans;
}

template CRS<double> CRS<double>::copy();
template CRS<float> CRS<float>::copy();

// copy monolish CRS
template <typename T> void CRS<T>::operator=(const CRS<T> &mat) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  val.resize(mat.get_nnz());
  col_ind.resize(mat.get_nnz());
  row_ptr.resize(mat.get_row() + 1);

  rowN = mat.get_row();
  colN = mat.get_col();
  nnz = mat.get_nnz();

#if MONOLISH_USE_GPU
  if(mat.get_device_mem_stat() == true){
    send();
  }
#endif

  internal::vcopy((get_row()+1), mat.row_ptr.data(), row_ptr.data(), get_device_mem_stat());
  internal::vcopy(get_nnz(), mat.col_ind.data(), col_ind.data(), get_device_mem_stat());
  internal::vcopy(get_nnz(), mat.val.data(), val.data(), get_device_mem_stat());

  logger.util_out();
}

template void CRS<double>::operator=(const CRS<double> &mat);
template void CRS<float>::operator=(const CRS<float> &mat);

} // namespace matrix
} // namespace monolish
