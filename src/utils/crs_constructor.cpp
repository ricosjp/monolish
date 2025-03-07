#include "../../include/monolish/common/monolish_logger.hpp"
#include "../../include/monolish/common/monolish_matrix.hpp"
#include "../../include/monolish/common/monolish_vector.hpp"
#include "../internal/monolish_internal.hpp"

namespace monolish {
namespace matrix {

// constructor ///
template <typename T>
CRS<T>::CRS(const size_t M, const size_t N, const size_t NNZ) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  set_row(M);
  set_col(N);

  row_ptr.resize(M + 1);
  col_ind.resize(NNZ);
  val_create_flag = true;
  resize(NNZ);
  logger.util_out();
}
template CRS<double>::CRS(const size_t M, const size_t N, const size_t NNZ);
template CRS<float>::CRS(const size_t M, const size_t N, const size_t NNZ);

template <typename T>
CRS<T>::CRS(const size_t M, const size_t N, const size_t NNZ, const int *rowptr,
            const int *colind, const T *value) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  set_row(M);
  set_col(N);

  row_ptr.resize(M + 1);
  col_ind.resize(NNZ);
  val_create_flag = true;
  resize(NNZ);
  std::copy(rowptr, rowptr + (M + 1), row_ptr.begin());
  std::copy(colind, colind + NNZ, col_ind.begin());
  std::copy(value, value + NNZ, data());
  compute_hash();
  logger.util_out();
}
template CRS<double>::CRS(const size_t M, const size_t N, const size_t NNZ,
                          const int *rowptr, const int *colind,
                          const double *value);
template CRS<float>::CRS(const size_t M, const size_t N, const size_t NNZ,
                         const int *rowptr, const int *colind,
                         const float *value);

template <typename T>
CRS<T>::CRS(const size_t M, const size_t N, const size_t NNZ, const int *rowptr,
            const int *colind, const T *value, const size_t origin) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  set_row(M);
  set_col(N);

  row_ptr.resize(M + 1);
  col_ind.resize(NNZ);
  val_create_flag = true;
  resize(NNZ);
  std::copy(rowptr, rowptr + (M + 1), row_ptr.begin());
  std::copy(colind, colind + NNZ, col_ind.begin());
  std::copy(value, value + NNZ, data());

#pragma omp parallel for
  for (size_t i = 0; i < NNZ; i++) {
    col_ind[i] -= origin;
  }

  compute_hash();
  logger.util_out();
}
template CRS<double>::CRS(const size_t M, const size_t N, const size_t NNZ,
                          const int *rowptr, const int *colind,
                          const double *value, const size_t origin);
template CRS<float>::CRS(const size_t M, const size_t N, const size_t NNZ,
                         const int *rowptr, const int *colind,
                         const float *value, const size_t origin);

template <typename T>
CRS<T>::CRS(const size_t M, const size_t N, const std::vector<int> &rowptr,
            const std::vector<int> &colind, const std::vector<T> &value) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  set_row(M);
  set_col(N);

  row_ptr.resize(M + 1);
  col_ind.resize(value.size());
  val_create_flag = true;
  resize(value.size());
  std::copy(rowptr.data(), rowptr.data() + (M + 1), row_ptr.begin());
  std::copy(colind.data(), colind.data() + value.size(), col_ind.begin());
  std::copy(value.data(), value.data() + value.size(), data());
  compute_hash();
  logger.util_out();
}
template CRS<double>::CRS(const size_t M, const size_t N,
                          const std::vector<int> &rowptr,
                          const std::vector<int> &colind,
                          const std::vector<double> &value);
template CRS<float>::CRS(const size_t M, const size_t N,
                         const std::vector<int> &rowptr,
                         const std::vector<int> &colind,
                         const std::vector<float> &value);

template <typename T>
CRS<T>::CRS(const size_t M, const size_t N, const std::vector<int> &rowptr,
            const std::vector<int> &colind, const std::shared_ptr<T> &value_) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  set_row(M);
  set_col(N);

  row_ptr.resize(M + 1);
  col_ind.resize(colind.size());
  val_create_flag = false;
  this->val = value_;
  std::copy(rowptr.data(), rowptr.data() + (M + 1), row_ptr.begin());
  std::copy(colind.data(), colind.data() + col_ind.size(), col_ind.begin());
  compute_hash();
  logger.util_out();
}
template CRS<double>::CRS(const size_t M, const size_t N,
                          const std::vector<int> &rowptr,
                          const std::vector<int> &colind,
                          const std::shared_ptr<double> &value_);
template CRS<float>::CRS(const size_t M, const size_t N,
                         const std::vector<int> &rowptr,
                         const std::vector<int> &colind,
                         const std::shared_ptr<float> &value_);

template <typename T>
CRS<T>::CRS(const size_t M, const size_t N, const std::vector<int> &rowptr,
            const std::vector<int> &colind, const vector<T> &value) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  set_row(M);
  set_col(N);

  row_ptr.resize(M + 1);
  col_ind.resize(value.size());
  val_create_flag = true;
  resize(value.size());
  std::copy(rowptr.data(), rowptr.data() + (M + 1), row_ptr.begin());
  std::copy(colind.data(), colind.data() + value.size(), col_ind.begin());
  std::copy(value.data(), value.data() + value.size(), begin());

  if (value.get_device_mem_stat() == true) {
#if MONOLISH_USE_NVIDIA_GPU
    send();
    T *vald = data();
    const T *data = value.data();
#pragma omp target teams distribute parallel for
    for (size_t i = 0; i < get_nnz(); i++) {
      vald[i] = data[i];
    }
#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  }

  compute_hash();
  logger.util_out();
}

template CRS<double>::CRS(const size_t M, const size_t N,
                          const std::vector<int> &rowptr,
                          const std::vector<int> &colind,
                          const vector<double> &value);
template CRS<float>::CRS(const size_t M, const size_t N,
                         const std::vector<int> &rowptr,
                         const std::vector<int> &colind,
                         const vector<float> &value);

// copy constructor
template <typename T> CRS<T>::CRS(const CRS<T> &mat) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  val_create_flag = true;
  resize(mat.get_nnz());
  col_ind.resize(mat.col_ind.size());
  row_ptr.resize(mat.row_ptr.size());

  rowN = mat.get_row();
  colN = mat.get_col();

#if MONOLISH_USE_NVIDIA_GPU
  if (mat.get_device_mem_stat()) {
    send();
    internal::vcopy(mat.row_ptr.size(), mat.row_ptr.data(), row_ptr.data(),
                    true);
    internal::vcopy(mat.col_ind.size(), mat.col_ind.data(), col_ind.data(),
                    true);
    internal::vcopy(mat.get_nnz(), mat.begin(), begin(), true);
  }
#endif

  internal::vcopy(mat.row_ptr.size(), mat.row_ptr.data(), row_ptr.data(),
                  false);
  internal::vcopy(mat.col_ind.size(), mat.col_ind.data(), col_ind.data(),
                  false);
  internal::vcopy(mat.get_nnz(), mat.begin(), begin(), false);

  compute_hash();
  logger.util_out();
}
template CRS<double>::CRS(const CRS<double> &mat);
template CRS<float>::CRS(const CRS<float> &mat);

// initialization constructor
template <typename T> CRS<T>::CRS(const CRS<T> &mat, T value) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  val_create_flag = true;
  resize(mat.get_nnz());
  col_ind.resize(mat.get_nnz());
  row_ptr.resize(mat.get_row() + 1);

  rowN = mat.get_row();
  colN = mat.get_col();

#if MONOLISH_USE_NVIDIA_GPU
  if (mat.get_device_mem_stat()) {
    send();
    internal::vcopy(mat.row_ptr.size(), mat.row_ptr.data(), row_ptr.data(),
                    true);
    internal::vcopy(mat.col_ind.size(), mat.col_ind.data(), col_ind.data(),
                    true);
    internal::vbroadcast(mat.get_nnz(), value, data(), true);
  }
#endif

  internal::vcopy(mat.row_ptr.size(), mat.row_ptr.data(), row_ptr.data(),
                  false);
  internal::vcopy(mat.col_ind.size(), mat.col_ind.data(), col_ind.data(),
                  false);
  internal::vbroadcast(mat.get_nnz(), value, data(), false);

  compute_hash();
  logger.util_out();
}
template CRS<double>::CRS(const CRS<double> &mat, double value);
template CRS<float>::CRS(const CRS<float> &mat, float value);
} // namespace matrix
} // namespace monolish
