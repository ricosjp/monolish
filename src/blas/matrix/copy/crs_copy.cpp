#include "../../../../include/monolish_blas.hpp"
#include "../../../monolish_internal.hpp"

namespace monolish {
namespace matrix {

// copy
template <typename T> CRS<T> CRS<T>::copy() {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  if (get_device_mem_stat()) {
    nonfree_recv();
  } // gpu copy

  CRS<T> tmp;
  std::copy(row_ptr.data(), row_ptr.data() + (get_row() + 1),
            tmp.row_ptr.begin());
  std::copy(col_ind.data(), col_ind.data() + get_nnz(), tmp.col_ind.begin());
  std::copy(val.data(), val.data() + get_nnz(), tmp.val.begin());
  tmp.rowN = get_row();
  tmp.colN = get_col();
  tmp.nnz = get_nnz();
  if (get_device_mem_stat()) {
    tmp.send();
  } // gpu copy

  logger.util_out();
  return tmp;
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

  size_t N = get_row();
  size_t NNZ = get_nnz();

  // gpu copy and recv
  if (mat.get_device_mem_stat()) {
    send();
    T *vald = val.data();
    int *cold = col_ind.data();
    int *rowd = row_ptr.data();

    const T *Mvald = mat.val.data();
    const int *Mcold = mat.col_ind.data();
    const int *Mrowd = mat.row_ptr.data();

#if USE_GPU

#pragma acc data present(rowd [0:N + 1], Mrowd [0:N + 1])
#pragma acc parallel
#pragma acc loop independent
    for (size_t i = 0; i < N + 1; i++) {
      rowd[i] = Mrowd[i];
    }

#pragma acc data present(vald [0:nnz], cold [0:nnz], Mvald [0:nnz],            \
                         Mcold [0:nnz])
#pragma acc parallel
#pragma acc loop independent
    for (size_t i = 0; i < NNZ; i++) {
      cold[i] = Mcold[i];
      vald[i] = Mvald[i];
    }

    nonfree_recv();
#endif
  } else {
    std::copy(mat.row_ptr.data(), mat.row_ptr.data() + (get_row() + 1),
              row_ptr.begin());
    std::copy(mat.col_ind.data(), mat.col_ind.data() + get_nnz(),
              col_ind.begin());
    std::copy(mat.val.data(), mat.val.data() + get_nnz(), val.begin());
  }

  logger.util_out();
}

template void CRS<double>::operator=(const CRS<double> &mat);
template void CRS<float>::operator=(const CRS<float> &mat);

} // namespace matrix
} // namespace monolish
