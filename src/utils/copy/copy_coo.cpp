#include "../../../include/monolish/common/monolish_common.hpp"
#include "../../../include/monolish/common/monolish_dense.hpp"
#include "../../../include/monolish/common/monolish_logger.hpp"
#include "../../../include/monolish/common/monolish_matrix.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {
namespace matrix {

template <typename T> void COO<T>::operator=(const matrix::COO<T> &mat) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  // err
  assert(monolish::util::is_same_size(*this, mat));
  assert(monolish::util::is_same_structure(*this, mat));
  assert(monolish::util::is_same_device_mem_stat(*this, mat));

  // value copy
  internal::vcopy(get_nnz(), mat.begin(), begin(), get_device_mem_stat());

  logger.util_out();
}
template void COO<double>::operator=(const matrix::COO<double> &mat);
template void COO<float>::operator=(const matrix::COO<float> &mat);

template <typename T>
void COO<T>::set_ptr(const size_t rN, const size_t cN,
                     const std::vector<int> &r, const std::vector<int> &c,
                     const size_t vsize, const T *v) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  col_index = c;
  row_index = r;
  resize(vsize);

  internal::vcopy(get_nnz(), v, begin(), false);

  rowN = rN;
  colN = cN;
  logger.util_out();
}
template void COO<double>::set_ptr(const size_t rN, const size_t cN,
                                   const std::vector<int> &r,
                                   const std::vector<int> &c,
                                   const size_t vsize, const double *v);
template void COO<float>::set_ptr(const size_t rN, const size_t cN,
                                  const std::vector<int> &r,
                                  const std::vector<int> &c, const size_t vsize,
                                  const float *v);

template <typename T>
void COO<T>::set_ptr(const size_t rN, const size_t cN,
                     const std::vector<int> &r, const std::vector<int> &c,
                     const size_t vsize, const T v) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  col_index = c;
  row_index = r;
  resize(vsize);

  internal::vbroadcast(get_nnz(), v, begin(), false);

  rowN = rN;
  colN = cN;
  logger.util_out();
}
template void COO<double>::set_ptr(const size_t rN, const size_t cN,
                                   const std::vector<int> &r,
                                   const std::vector<int> &c,
                                   const size_t vsize, const double v);
template void COO<float>::set_ptr(const size_t rN, const size_t cN,
                                  const std::vector<int> &r,
                                  const std::vector<int> &c, const size_t vsize,
                                  const float v);

template <typename T>
void COO<T>::set_ptr(const size_t rN, const size_t cN,
                     const std::vector<int> &r, const std::vector<int> &c,
                     const std::vector<T> &v) {
  set_ptr(rN, cN, r, c, v.size(), v.data());
}
template void COO<double>::set_ptr(const size_t rN, const size_t cN,
                                   const std::vector<int> &r,
                                   const std::vector<int> &c,
                                   const std::vector<double> &v);
template void COO<float>::set_ptr(const size_t rN, const size_t cN,
                                  const std::vector<int> &r,
                                  const std::vector<int> &c,
                                  const std::vector<float> &v);

} // namespace matrix
} // namespace monolish
