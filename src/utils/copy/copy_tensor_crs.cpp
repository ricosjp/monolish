#include "../../../include/monolish/common/monolish_common.hpp"
#include "../../../include/monolish/common/monolish_dense.hpp"
#include "../../../include/monolish/common/monolish_logger.hpp"
#include "../../../include/monolish/common/monolish_matrix.hpp"
#include "../../../include/monolish/common/monolish_tensor.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {
namespace tensor {

template <typename T> void tensor_CRS<T>::operator=(const tensor_CRS<T> &tens) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  // err
  assert(monolish::util::is_same_size(*this, tens));
  assert(monolish::util::is_same_structure(*this, tens));
  assert(monolish::util::is_same_device_mem_stat(*this, tens));
  val_create_flag = true;

  // value copy
  internal::vcopy(get_nnz(), tens.begin(), begin(), get_device_mem_stat());

  logger.util_out();
}
template void tensor_CRS<double>::operator=(const tensor_CRS<double> &tens);
template void tensor_CRS<float>::operator=(const tensor_CRS<float> &tens);

template <typename T>
void tensor_CRS<T>::set_ptr(const std::vector<size_t> &shape,
                            const std::vector<std::vector<int>> &rowptrs,
                            const std::vector<std::vector<int>> &colinds,
                            const size_t vsize, const T *value) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  col_inds = colinds;
  row_ptrs = rowptrs;
  val_create_flag = true;
  resize(vsize);

  internal::vcopy(get_nnz(), value, begin(), false);

  set_shape(shape);

  compute_hash();
  logger.util_out();
}
template void
tensor_CRS<double>::set_ptr(const std::vector<size_t> &shape,
                            const std::vector<std::vector<int>> &rowptr,
                            const std::vector<std::vector<int>> &colind,
                            const size_t vsize, const double *value);
template void
tensor_CRS<float>::set_ptr(const std::vector<size_t> &shape,
                           const std::vector<std::vector<int>> &rowptr,
                           const std::vector<std::vector<int>> &colind,
                           const size_t vsize, const float *value);

template <typename T>
void tensor_CRS<T>::set_ptr(const std::vector<size_t> &shape,
                            const std::vector<std::vector<int>> &rowptrs,
                            const std::vector<std::vector<int>> &colinds,
                            const size_t vsize, const T value) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  col_inds = colinds;
  row_ptrs = rowptrs;
  val_create_flag = true;
  resize(vsize);

  internal::vbroadcast(get_nnz(), value, begin(), false);

  set_shape(shape);

  compute_hash();
  logger.util_out();
}
template void
tensor_CRS<double>::set_ptr(const std::vector<size_t> &shape,
                            const std::vector<std::vector<int>> &rowptr,
                            const std::vector<std::vector<int>> &colind,
                            const size_t vsize, const double value);
template void
tensor_CRS<float>::set_ptr(const std::vector<size_t> &shape,
                           const std::vector<std::vector<int>> &rowptr,
                           const std::vector<std::vector<int>> &colind,
                           const size_t vsize, const float value);

template <typename T>
void tensor_CRS<T>::set_ptr(const std::vector<size_t> &shape,
                            const std::vector<std::vector<int>> &rowptrs,
                            const std::vector<std::vector<int>> &colinds,
                            const std::vector<T> &value) {
  set_ptr(shape, rowptrs, colinds, value.size(), value.data());
}
template void
tensor_CRS<double>::set_ptr(const std::vector<size_t> &shape,
                            const std::vector<std::vector<int>> &rowptr,
                            const std::vector<std::vector<int>> &colind,
                            const std::vector<double> &value);
template void
tensor_CRS<float>::set_ptr(const std::vector<size_t> &shape,
                           const std::vector<std::vector<int>> &rowptr,
                           const std::vector<std::vector<int>> &colind,
                           const std::vector<float> &value);

} // namespace tensor
} // namespace monolish
