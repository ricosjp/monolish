#include "../../include/monolish/common/monolish_dense.hpp"
#include "../../include/monolish/common/monolish_logger.hpp"
#include "../../include/monolish/common/monolish_matrix.hpp"
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
  gpu_status = false;
  row_index.resize(NNZ);
  col_index.resize(NNZ);

  vad_create_flag = true;
  resize(NNZ);

  std::copy(row, row + NNZ, row_index.begin());
  std::copy(col, col + NNZ, col_index.begin());
  std::copy(value, value + NNZ, data());
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
  gpu_status = false;
  row_index.resize(NNZ);
  col_index.resize(NNZ);

  vad_create_flag = true;
  resize(NNZ);

  std::copy(row, row + NNZ, row_index.begin());
  std::copy(col, col + NNZ, col_index.begin());
  std::copy(value, value + NNZ, data());

#pragma omp parallel for
  for (size_t i = 0; i < NNZ; i++) {
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
  gpu_status = false;
  row_index.resize(coo.get_nnz());
  col_index.resize(coo.get_nnz());
  vad_create_flag = true;
  resize(coo.get_nnz());
  std::copy(coo.row_index.data(), coo.row_index.data() + coo.get_nnz(),
            row_index.begin());
  std::copy(coo.col_index.data(), coo.col_index.data() + coo.get_nnz(),
            col_index.begin());
  std::copy(coo.data(), coo.data() + coo.get_nnz(), data());
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
  gpu_status = false;

  row_index.resize(coo.get_nnz());
  std::copy(coo.row_index.data(), coo.row_index.data() + coo.get_nnz(),
            row_index.begin());

  col_index.resize(coo.get_nnz());
  std::copy(coo.col_index.data(), coo.col_index.data() + coo.get_nnz(),
            col_index.begin());

  vad_create_flag = true;
  resize(coo.get_nnz());
  internal::vbroadcast(coo.get_nnz(), value, data(), false);

  logger.util_out();
}
template COO<double>::COO(const matrix::COO<double> &coo, double value);
template COO<float>::COO(const matrix::COO<float> &coo, float value);

} // namespace matrix
} // namespace monolish
