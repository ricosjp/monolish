#include "../../../include/common/monolish_dense.hpp"
#include "../../../include/common/monolish_logger.hpp"
#include "../../../include/common/monolish_matrix.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {
namespace matrix {

template <typename T> T Dense<T>::at(const size_t i, const size_t j) {
  if (get_device_mem_stat()) {
    throw std::runtime_error("at() Error, GPU vector cant use operator[]");
  }
  if (get_row() < i) {
    throw std::runtime_error("at() Error, A.row < i");
  }
  if (get_col() < j) {
    throw std::runtime_error("at() Error, A.col < j");
  }
  return val[get_col() * i + j];
}
template double Dense<double>::at(const size_t i, const size_t j);
template float Dense<float>::at(const size_t i, const size_t j);

template <typename T> T Dense<T>::at(const size_t i, const size_t j) const {
  if (get_device_mem_stat()) {
    throw std::runtime_error("at() Error, GPU vector cant use operator[]");
  }
  if (get_row() < i) {
    throw std::runtime_error("at() Error, A.row < i");
  }
  if (get_col() < j) {
    throw std::runtime_error("at() Error, A.col < j");
  }
  return val[get_col() * i + j];
}
template double Dense<double>::at(const size_t i, const size_t j) const;
template float Dense<float>::at(const size_t i, const size_t j) const;

// insert //

template <typename T>
void Dense<T>::insert(const size_t i, const size_t j, const T Val) {
  if (get_device_mem_stat()) {
    throw std::runtime_error("insert() Error, GPU vector cant use operator[]");
  }
  if (get_row() < i) {
    throw std::runtime_error("insert() Error, A.row < i");
  }
  if (get_col() < j) {
    throw std::runtime_error("insert() Error, A.col < j");
  }
  val[get_col() * i + j] = Val;
}
template void Dense<double>::insert(const size_t i, const size_t j,
                                    const double Val);
template void Dense<float>::insert(const size_t i, const size_t j,
                                   const float Val);

}
}
