#include "../../../include/monolish/common/monolish_common.hpp"
#include "../../../include/monolish/common/monolish_dense.hpp"
#include "../../../include/monolish/common/monolish_logger.hpp"
#include "../../../include/monolish/common/monolish_matrix.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {
namespace matrix {

template <typename T> void CRS<T>::operator=(const CRS<T> &mat) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  // err
  assert(monolish::util::is_same_size(*this, mat));
  assert(monolish::util::is_same_structure(*this, mat));
  assert(monolish::util::is_same_device_mem_stat(*this, mat));
  val_create_flag = true;

  // value copy
  internal::vcopy(get_nnz(), mat.begin(), begin(), get_device_mem_stat());

  logger.util_out();
}
template void CRS<double>::operator=(const CRS<double> &mat);
template void CRS<float>::operator=(const CRS<float> &mat);

template <typename T>
void CRS<T>::set_ptr(const size_t M, const size_t N,
                     const std::vector<int> &rowptr,
                     const std::vector<int> &colind, const size_t vsize,
                     const T *value) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  col_ind = colind;
  row_ptr = rowptr;
  val_create_flag = true;
  resize(vsize);

  internal::vcopy(get_nnz(), value, begin(), false);

  rowN = M;
  colN = N;

  compute_hash();
  logger.util_out();
}
template void CRS<double>::set_ptr(const size_t M, const size_t N,
                                   const std::vector<int> &rowptr,
                                   const std::vector<int> &colind,
                                   const size_t vsize, const double *value);
template void CRS<float>::set_ptr(const size_t M, const size_t N,
                                  const std::vector<int> &rowptr,
                                  const std::vector<int> &colind,
                                  const size_t vsize, const float *value);

template <typename T>
void CRS<T>::set_ptr(const size_t M, const size_t N,
                     const std::vector<int> &rowptr,
                     const std::vector<int> &colind, const size_t vsize,
                     const T value) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  col_ind = colind;
  row_ptr = rowptr;
  val_create_flag = true;
  resize(vsize);

  internal::vbroadcast(get_nnz(), value, begin(), false);

  rowN = M;
  colN = N;

  compute_hash();
  logger.util_out();
}
template void CRS<double>::set_ptr(const size_t M, const size_t N,
                                   const std::vector<int> &rowptr,
                                   const std::vector<int> &colind,
                                   const size_t vsize, const double value);
template void CRS<float>::set_ptr(const size_t M, const size_t N,
                                  const std::vector<int> &rowptr,
                                  const std::vector<int> &colind,
                                  const size_t vsize, const float value);

template <typename T>
void CRS<T>::set_ptr(const size_t M, const size_t N,
                     const std::vector<int> &rowptr,
                     const std::vector<int> &colind,
                     const std::vector<T> &value) {
  set_ptr(M, N, rowptr, colind, value.size(), value.data());
}
template void CRS<double>::set_ptr(const size_t M, const size_t N,
                                   const std::vector<int> &rowptr,
                                   const std::vector<int> &colind,
                                   const std::vector<double> &value);
template void CRS<float>::set_ptr(const size_t M, const size_t N,
                                  const std::vector<int> &rowptr,
                                  const std::vector<int> &colind,
                                  const std::vector<float> &value);

} // namespace matrix
} // namespace monolish
