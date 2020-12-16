#include "../../include/common/monolish_logger.hpp"
#include "../../include/common/monolish_matrix.hpp"
#include "../../include/common/monolish_vector.hpp"
#include "../internal/monolish_internal.hpp"

#include <cassert>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>

namespace monolish {
namespace matrix {

// constructor ///
template <typename T>
CRS<T>::CRS(const size_t M, const size_t N, const size_t NNZ) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  rowN = M;
  colN = N;
  nnz = NNZ;
  gpu_status = false;
  row_ptr.resize(M + 1);
  col_ind.resize(nnz);
  val.resize(nnz);
  logger.util_out();
}
template CRS<double>::CRS(const size_t M, const size_t N, const size_t NNZ);
template CRS<float>::CRS(const size_t M, const size_t N, const size_t NNZ);

template <typename T>
CRS<T>::CRS(const size_t M, const size_t N, const size_t NNZ, const int *rowptr,
            const int *colind, const T *value) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  rowN = M;
  colN = N;
  nnz = NNZ;
  gpu_status = false;
  row_ptr.resize(M + 1);
  col_ind.resize(nnz);
  val.resize(nnz);
  std::copy(rowptr, rowptr + (M + 1), row_ptr.begin());
  std::copy(colind, colind + nnz, col_ind.begin());
  std::copy(value, value + nnz, val.begin());
  logger.util_out();
}
template CRS<double>::CRS(const size_t M, const size_t N, const size_t NNZ,
                          const int *rowptr, const int *colind,
                          const double *value);
template CRS<float>::CRS(const size_t M, const size_t N, const size_t NNZ,
                         const int *rowptr, const int *colind,
                         const float *value);

template <typename T>
CRS<T>::CRS(const size_t M, const size_t N, const std::vector<int> rowptr,
            const std::vector<int> colind, const std::vector<T> value) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  rowN = M;
  colN = N;
  nnz = value.size();
  gpu_status = false;
  row_ptr.resize(M + 1);
  col_ind.resize(nnz);
  val.resize(nnz);
  std::copy(rowptr.data(), rowptr.data() + (M + 1), row_ptr.begin());
  std::copy(colind.data(), colind.data() + nnz, col_ind.begin());
  std::copy(value.data(), value.data() + nnz, val.begin());
  logger.util_out();
}
template CRS<double>::CRS(const size_t M, const size_t N,
                          const std::vector<int> rowptr,
                          const std::vector<int> colind,
                          const std::vector<double> value);
template CRS<float>::CRS(const size_t M, const size_t N,
                         const std::vector<int> rowptr,
                         const std::vector<int> colind,
                         const std::vector<float> value);

// convert ///
//
// copy constractor
template <typename T> CRS<T>::CRS(const CRS<T> &mat) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  val.resize(mat.get_nnz());
  col_ind.resize(mat.get_nnz());
  row_ptr.resize(mat.get_row() + 1);

  rowN = mat.get_row();
  colN = mat.get_col();
  nnz = mat.get_nnz();

  // gpu copy and recv
  if (mat.get_device_mem_stat()) {
    send();

#if MONOLISH_USE_GPU
    size_t N = get_row();
    size_t NNZ = get_nnz();
    T *vald = val.data();
    int *cold = col_ind.data();
    int *rowd = row_ptr.data();

    const T *Mvald = mat.val.data();
    const int *Mcold = mat.col_ind.data();
    const int *Mrowd = mat.row_ptr.data();

#pragma omp target teams distribute parallel for
    for (size_t i = 0; i < N + 1; i++) {
      rowd[i] = Mrowd[i];
    }

#pragma omp target teams distribute parallel for
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
template CRS<double>::CRS(const CRS<double> &mat);
template CRS<float>::CRS(const CRS<float> &mat);

template <typename T> void CRS<T>::convert(COO<T> &coo) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  // todo coo err check (only square)

  rowN = coo.get_row();
  colN = coo.get_col();
  nnz = coo.get_nnz();

  val = coo.val;
  col_ind = coo.col_index;

  // todo not inplace now
  row_ptr.resize(get_row() + 1, 0.0);

  row_ptr[0] = 0;
  size_t c_row = 0;
  for (size_t i = 0; i < coo.get_nnz(); i++) {

    if ((int)c_row == coo.row_index[i]) {
      row_ptr[c_row + 1] = i + 1;
    } else {
      c_row = c_row + 1;
      row_ptr[c_row + 1] = i + 1;
    }
  }
  logger.util_out();
}
template void CRS<double>::convert(COO<double> &coo);
template void CRS<float>::convert(COO<float> &coo);

template <typename T> void CRS<T>::print_all() {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  for (size_t i = 0; i < get_row(); i++) {
    for (size_t j = (size_t)row_ptr[i]; j < (size_t)row_ptr[i + 1]; j++) {
      std::cout << i + 1 << " " << col_ind[j] + 1 << " " << val[j] << std::endl;
    }
  }

  logger.util_out();
}
template void CRS<double>::print_all();
template void CRS<float>::print_all();

template <typename T> bool CRS<T>::operator==(const CRS<T> &mat) const {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  if (get_device_mem_stat()) {
    throw std::runtime_error("Error, GPU CRS cant use operator==");
  }

  if (get_row() != mat.get_row()) {
    return false;
  }
  if (get_col() != mat.get_col()) {
    return false;
  }

  if (get_device_mem_stat() != mat.get_device_mem_stat()) {
    return false;
  }

  if (get_device_mem_stat() == true) {
    if( !(internal::vequal(get_nnz(), val.data(), mat.val.data(), true)) ){
      return false;
    }
    if( !(internal::vequal(get_nnz(), col_ind.data(), mat.col_ind.data(), true)) ){
      return false;
    }
    if( !(internal::vequal(get_nnz(), row_ptr.data(), mat.row_ptr.data(), true)) ){
      return false;
    }
  }

  if( !(internal::vequal(get_nnz(), val.data(), mat.val.data(), false)) ){
    return false;
  }
  if( !(internal::vequal(get_nnz(), col_ind.data(), mat.col_ind.data(), false)) ){
    return false;
  }
  if( !(internal::vequal(get_nnz(), row_ptr.data(), mat.row_ptr.data(), false)) ){
    return false;
  }

  logger.util_out();
  return true;
}
template bool CRS<double>::operator==(const CRS<double> &mat) const;
template bool CRS<float>::operator==(const CRS<float> &mat) const;

template <typename T> bool CRS<T>::operator!=(const CRS<T> &mat) const {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  if (get_device_mem_stat()) {
    throw std::runtime_error("Error, GPU CRS cant use operator!=");
  }

  if (get_row() != mat.get_row()) {
    return true;
  }
  if (get_col() != mat.get_col()) {
    return true;
  }

  if (get_device_mem_stat() != mat.get_device_mem_stat()) {
    return true;
  }

  if (get_device_mem_stat() == true) {
    if( internal::vequal(get_nnz(), val.data(), mat.val.data(), true) ){
      return false;
    }
    if( internal::vequal(get_nnz(), col_ind.data(), mat.col_ind.data(), true) ){
      return false;
    }
    if( internal::vequal(get_nnz(), row_ptr.data(), mat.row_ptr.data(), true) ){
      return false;
    }
  }

  if( internal::vequal(get_nnz(), val.data(), mat.val.data(), false) ){
    return false;
  }
  if( internal::vequal(get_nnz(), col_ind.data(), mat.col_ind.data(), false) ){
    return false;
  }
  if( internal::vequal(get_nnz(), row_ptr.data(), mat.row_ptr.data(), false) ){
    return false;
  }

  logger.util_out();
  return true;
}
template bool CRS<double>::operator!=(const CRS<double> &mat) const;
template bool CRS<float>::operator!=(const CRS<float> &mat) const;
} // namespace matrix
} // namespace monolish
