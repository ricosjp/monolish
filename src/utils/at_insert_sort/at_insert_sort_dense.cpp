#include "../../../include/monolish/common/monolish_dense.hpp"
#include "../../../include/monolish/common/monolish_logger.hpp"
#include "../../../include/monolish/common/monolish_matrix.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {
namespace matrix {

template <typename T> T Dense<T>::at(const size_t i) const {
  if (get_device_mem_stat()) {
    throw std::runtime_error("at() Error, GPU vector cant use operator[]");
  }

  assert(i < get_nnz());
  assert(first + i < get_alloc_nnz());

  return data()[first + i];
}
template double Dense<double>::at(const size_t i) const;
template float Dense<float>::at(const size_t i) const;

template <typename T> T Dense<T>::at(const size_t i, const size_t j) const {
  if (get_device_mem_stat()) {
    throw std::runtime_error("at() Error, GPU vector cant use operator[]");
  }

  assert(i < get_row());
  assert(j < get_col());

  return at(get_col() * i + j);
}
template double Dense<double>::at(const size_t i, const size_t j) const;
template float Dense<float>::at(const size_t i, const size_t j) const;

// insert //
template <typename T> void Dense<T>::insert(const size_t i, const T Val) {
  if (get_device_mem_stat()) {
    throw std::runtime_error("insert() Error, GPU vector cant use operator[]");
  }

  assert(i < get_nnz());
  assert(first + i < get_alloc_nnz());

  data()[first + i] = Val;
}
template void Dense<double>::insert(const size_t i, const double Val);
template void Dense<float>::insert(const size_t i, const float Val);

template <typename T>
void Dense<T>::insert(const size_t i, const size_t j, const T Val) {
  if (get_device_mem_stat()) {
    throw std::runtime_error("insert() Error, GPU vector cant use operator[]");
  }

  assert(i < get_row());
  assert(j < get_col());

  insert(get_col() * i + j, Val);
}
template void Dense<double>::insert(const size_t i, const size_t j,
                                    const double Val);
template void Dense<float>::insert(const size_t i, const size_t j,
                                   const float Val);

} // namespace matrix
} // namespace monolish
