#include "../../include/common/monolish_dense.hpp"
#include "../../include/common/monolish_logger.hpp"
#include "../../include/common/monolish_matrix.hpp"
#include "../internal/monolish_internal.hpp"

// todo: kill cerr

namespace monolish {
namespace matrix {

// matrix constructor

template <typename T>
COO<T>::COO(const size_t M, const size_t N, const size_t NNZ, const int *row,
            const int *col, const T *value) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  rowN = M;
  colN = N;
  nnz = NNZ;
  gpu_status = false;
  row_index.resize(nnz);
  col_index.resize(nnz);
  val.resize(nnz);

  std::copy(row, row + nnz, row_index.begin());
  std::copy(col, col + nnz, col_index.begin());
  std::copy(value, value + nnz, val.begin());
  logger.util_out();
}
template COO<double>::COO(const size_t M, const size_t N, const size_t NNZ,
                          const int *row, const int *col, const double *value);
template COO<float>::COO(const size_t M, const size_t N, const size_t NNZ,
                         const int *row, const int *col, const float *value);

template <typename T>
COO<T>::COO(const size_t M, const size_t N, const size_t NNZ, const int *row,
            const int *col, const T *value, const size_t origin) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  rowN = M;
  colN = N;
  nnz = NNZ;
  gpu_status = false;
  row_index.resize(nnz);
  col_index.resize(nnz);
  val.resize(nnz);

  std::copy(row, row + nnz, row_index.begin());
  std::copy(col, col + nnz, col_index.begin());
  std::copy(value, value + nnz, val.begin());

#pragma omp parallel for
  for (size_t i = 0; i < nnz; i++) {
    row_index[i] -= origin;
    col_index[i] -= origin;
  }
  logger.util_out();
}
template COO<double>::COO(const size_t M, const size_t N, const size_t NNZ,
                          const int *row, const int *col, const double *value,
                          const size_t origin);
template COO<float>::COO(const size_t M, const size_t N, const size_t NNZ,
                         const int *row, const int *col, const float *value,
                         const size_t origin);

// copy constructor
template <typename T> COO<T>::COO(const matrix::COO<T> &coo) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  rowN = coo.get_row();
  colN = coo.get_col();
  nnz = coo.get_nnz();
  gpu_status = false;
  row_index.resize(nnz);
  col_index.resize(nnz);
  val.resize(nnz);
  std::copy(coo.row_index.data(), coo.row_index.data() + nnz,
            row_index.begin());
  std::copy(coo.col_index.data(), coo.col_index.data() + nnz,
            col_index.begin());
  std::copy(coo.val.data(), coo.val.data() + nnz, val.begin());
  logger.util_out();
}
template COO<double>::COO(const matrix::COO<double> &coo);
template COO<float>::COO(const matrix::COO<float> &coo);

// initializaion constructor
template <typename T> COO<T>::COO(const matrix::COO<T> &coo, T value) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  rowN = coo.get_row();
  colN = coo.get_col();
  nnz = coo.get_nnz();
  gpu_status = false;

  row_index.resize(nnz);
  std::copy(coo.row_index.data(), coo.row_index.data() + nnz,
            row_index.begin());

  col_index.resize(nnz);
  std::copy(coo.col_index.data(), coo.col_index.data() + nnz,
            col_index.begin());

  val.resize(nnz);
  for (auto i = decltype(nnz){0}; i < nnz; i++) {
    val[i] = value;
  }

  logger.util_out();
}
template COO<double>::COO(const matrix::COO<double> &coo, double value);
template COO<float>::COO(const matrix::COO<float> &coo, float value);

} // namespace matrix
} // namespace monolish
