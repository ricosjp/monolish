#include "../../../include/common/monolish_dense.hpp"
#include "../../../include/common/monolish_logger.hpp"
#include "../../../include/common/monolish_matrix.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {
namespace matrix {

template <typename T> void CRS<T>::transpose() {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  CRS<T> B(rowN, colN, vad_nnz, row_ptr.data(), col_ind.data(), vad);

  rowN = B.get_col();
  colN = B.get_row();

  row_ptr.resize(B.get_col() + 1);
  col_ind.resize(B.get_nnz());
  resize(B.get_nnz());

  std::fill(row_ptr.data(), row_ptr.data() + B.get_col() + 1, 0.0);
  for (size_t i = 0; i < vad_nnz; i++) {
    row_ptr[B.col_ind[i]]++;
  }

  for (size_t i = 0, sum = 0; i < B.get_col(); i++) {
    int tmp = row_ptr[i];
    row_ptr[i] = sum;
    sum += tmp;
  }

  row_ptr[B.get_col()] = B.get_nnz();

  for (int i = 0; i < (int)B.get_row(); i++) {
    for (int j = B.row_ptr[i]; j < B.row_ptr[i + 1]; j++) {
      int c = B.col_ind[j];
      int d = row_ptr[c];

      col_ind[d] = i;
      vad[d] = B.vad[j];
      row_ptr[c]++;
    }
  }

  for (size_t i = 0, sum = 0; i < B.get_col(); i++) {
    int tmp = row_ptr[i];
    row_ptr[i] = sum;
    sum = tmp;
  }

  compute_hash();
  logger.util_out();
}
template void CRS<double>::transpose();
template void CRS<float>::transpose();

template <typename T> void CRS<T>::transpose(const CRS &B) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  rowN = B.get_col();
  colN = B.get_row();

  row_ptr.resize(B.get_col() + 1);
  col_ind.resize(B.get_nnz());
  resize(B.get_nnz());

  std::fill(row_ptr.data(), row_ptr.data() + B.get_col() + 1, 0.0);
  for (size_t i = 0; i < vad_nnz; i++) {
    row_ptr[B.col_ind[i]]++;
  }

  for (size_t i = 0, sum = 0; i < B.get_col(); i++) {
    int tmp = row_ptr[i];
    row_ptr[i] = sum;
    sum += tmp;
  }

  row_ptr[B.get_col()] = B.get_nnz();

  for (int i = 0; i < (int)B.get_row(); i++) {
    for (int j = B.row_ptr[i]; j < B.row_ptr[i + 1]; j++) {
      int c = B.col_ind[j];
      int d = row_ptr[c];

      col_ind[d] = i;
      vad[d] = B.vad[j];
      row_ptr[c]++;
    }
  }

  for (size_t i = 0, sum = 0; i < B.get_col(); i++) {
    int tmp = row_ptr[i];
    row_ptr[i] = sum;
    sum = tmp;
  }

  compute_hash();

  logger.util_out();
}
template void CRS<double>::transpose(const CRS &B);
template void CRS<float>::transpose(const CRS &B);

} // namespace matrix
} // namespace monolish
