#include "../../../include/monolish/common/monolish_dense.hpp"
#include "../../../include/monolish/common/monolish_logger.hpp"
#include "../../../include/monolish/common/monolish_matrix.hpp"
#include "../../../include/monolish/common/monolish_tensor.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {
namespace tensor {

template <typename T> T tensor_Dense<T>::at(const size_t pos) const {
  if(get_device_mem_stat()) {
    throw std::runtime_error("at() Error, GPU vector cant use operator[]");
  }

  assert(pos <= get_nnz());

  return data()[pos];
}
template double tensor_Dense<double>::at(const size_t pos) const;
template float tensor_Dense<float>::at(const size_t pos) const;

template <typename T>
T tensor_Dense<T>::at(const std::vector<size_t> &pos) const {
  if (get_device_mem_stat()) {
    throw std::runtime_error("at() Error, GPU vector cant use operator[]");
  }

  auto ind = get_index(pos);

  return at(ind);
}
template double tensor_Dense<double>::at(const std::vector<size_t> &pos) const;
template float tensor_Dense<float>::at(const std::vector<size_t> &pos) const;

template <typename T>
void tensor_Dense<T>::insert(const size_t pos, const T Val) {
  if (get_device_mem_stat()) {
    throw std::runtime_error("at() Error, GPU vector cant use operator[]");
  }

  assert(pos < get_nnz());
  
  data()[pos] = Val;
}
template void tensor_Dense<double>::insert(const size_t pos, const double Val);
template void tensor_Dense<float>::insert(const size_t pos, const float Val);

template <typename T>
void tensor_Dense<T>::insert(const std::vector<size_t> &pos, const T Val) {
  if (get_device_mem_stat()) {
    throw std::runtime_error("at() Error, GPU vector cant use operator[]");
  }

  auto ind = get_index(pos);

  insert(ind, Val);
}
template void tensor_Dense<double>::insert(const std::vector<size_t> &pos,
                                           const double Val);
template void tensor_Dense<float>::insert(const std::vector<size_t> &pos,
                                          const float Val);

} // namespace tensor
} // namespace monolish
