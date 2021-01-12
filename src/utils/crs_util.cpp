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

// copy constructor
template <typename T> CRS<T>::CRS(const CRS<T> &mat) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  val.resize(mat.get_nnz());
  col_ind.resize(mat.get_nnz());
  row_ptr.resize(mat.get_row() + 1);

  rowN = mat.get_row();
  colN = mat.get_col();
  nnz = mat.get_nnz();

#if MONOLISH_USE_GPU
  if (mat.get_device_mem_stat()) {
    send();
    internal::vcopy(mat.row_ptr.size(), mat.row_ptr.data(), row_ptr.data(),
                    true);
    internal::vcopy(mat.col_ind.size(), mat.col_ind.data(), col_ind.data(),
                    true);
    internal::vcopy(mat.val.size(), mat.val.data(), val.data(), true);
  }
#endif

  internal::vcopy(mat.row_ptr.size(), mat.row_ptr.data(), row_ptr.data(),
                  false);
  internal::vcopy(mat.col_ind.size(), mat.col_ind.data(), col_ind.data(),
                  false);
  internal::vcopy(mat.val.size(), mat.val.data(), val.data(), false);

  logger.util_out();
}
template CRS<double>::CRS(const CRS<double> &mat);
template CRS<float>::CRS(const CRS<float> &mat);

// operator= (copy)
template <typename T> void CRS<T>::operator=(const CRS<T> &mat) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  // err
  if (get_row() != mat.get_row()) {
    throw std::runtime_error("error A.row != mat.row");
  }
  if (get_col() != mat.get_col()) {
    throw std::runtime_error("error A.col != mat.col");
  }
  if (get_nnz() != mat.get_nnz()) {
    throw std::runtime_error("error A.nnz != mat.nnz");
  }
  if (get_device_mem_stat() != mat.get_device_mem_stat()) {
    throw std::runtime_error("error get_device_mem_stat() is not same");
  }

  if (mat.get_device_mem_stat() == true) {
#if MONOLISH_USE_GPU
    internal::vcopy(mat.row_ptr.size(), mat.row_ptr.data(), row_ptr.data(),
                    true);
    internal::vcopy(mat.col_ind.size(), mat.col_ind.data(), col_ind.data(),
                    true);
    internal::vcopy(mat.val.size(), mat.val.data(), val.data(), true);
#endif
  } else {
    internal::vcopy(mat.row_ptr.size(), mat.row_ptr.data(), row_ptr.data(),
                    false);
    internal::vcopy(mat.col_ind.size(), mat.col_ind.data(), col_ind.data(),
                    false);
    internal::vcopy(mat.val.size(), mat.val.data(), val.data(), false);
  }

  logger.util_out();
}
template void CRS<double>::operator=(const CRS<double> &mat);
template void CRS<float>::operator=(const CRS<float> &mat);

// Utils
template <typename T> void CRS<T>::fill(T value) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  if (get_device_mem_stat() == true) {
#if MONOLISH_USE_GPU
#pragma omp target teams distribute parallel for
    for (size_t i = 0; i < get_nnz(); i++) {
      val[i] = value;
    }
#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  } else {
#pragma omp parallel for
    for (size_t i = 0; i < get_nnz(); i++) {
      val[i] = value;
    }
  }
  logger.util_out();
}
template void CRS<double>::fill(double value);
template void CRS<float>::fill(float value);

} // namespace matrix
} // namespace monolish
