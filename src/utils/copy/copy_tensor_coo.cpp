#include "../../../include/monolish/common/monolish_common.hpp"
#include "../../../include/monolish/common/monolish_dense.hpp"
#include "../../../include/monolish/common/monolish_logger.hpp"
#include "../../../include/monolish/common/monolish_matrix.hpp"
#include "../../../include/monolish/common/monolish_tensor.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {
namespace tensor {

template <typename T>
void tensor_COO<T>::operator=(const tensor::tensor_COO<T> &tens) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  // err
  assert(monolish::util::is_same_size(*this, tens));
  assert(monolish::util::is_same_structure(*this, tens));
  assert(monolish::util::is_same_device_mem_stat(*this, tens));

  // value copy
  internal::vcopy(get_nnz(), begin(), tens.begin(), get_device_mem_stat());

  logger.util_out();
}

template <typename T>
void tensor_COO<T>::set_ptr(const std::vector<size_t> &shape,
                            const std::vector<std::vector<size_t>> &index,
                            const size_t vsize, const T *v) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  val_create_flag = true;
  this->shape = shape;
  this->index = index;
  resize(vsize);

  internal::vcopy(get_nnz(), v, begin(), false);

  logger.util_out();
}
template void
tensor_COO<double>::set_ptr(const std::vector<size_t> &shape,
                            const std::vector<std::vector<size_t>> &index,
                            const size_t vsize, const double *v);
template void
tensor_COO<float>::set_ptr(const std::vector<size_t> &shape,
                           const std::vector<std::vector<size_t>> &index,
                           const size_t vsize, const float *v);

template <typename T>
void tensor_COO<T>::set_ptr(const std::vector<size_t> &shape,
                            const std::vector<std::vector<size_t>> &index,
                            const size_t vsize, const T v) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  val_create_flag = true;
  this->shape = shape;
  this->index = index;
  resize(vsize);

  internal::vbroadcast(get_nnz(), v, begin(), false);

  logger.util_out();
}
template void
tensor_COO<double>::set_ptr(const std::vector<size_t> &shape,
                            const std::vector<std::vector<size_t>> &index,
                            const size_t vsize, const double v);
template void
tensor_COO<float>::set_ptr(const std::vector<size_t> &shape,
                           const std::vector<std::vector<size_t>> &index,
                           const size_t vsize, const float v);

template <typename T>
void tensor_COO<T>::set_ptr(const std::vector<size_t> &shape,
                            const std::vector<std::vector<size_t>> &index,
                            const std::vector<T> &v) {
  set_ptr(shape, index, v.size(), v.data());
}
template void
tensor_COO<double>::set_ptr(const std::vector<size_t> &shape,
                            const std::vector<std::vector<size_t>> &index,
                            const std::vector<double> &y);
template void
tensor_COO<float>::set_ptr(const std::vector<size_t> &shape,
                           const std::vector<std::vector<size_t>> &index,
                           const std::vector<float> &v);

} // namespace tensor
} // namespace monolish
