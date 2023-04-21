#include "../../../include/monolish_blas.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {
namespace tensor {

// diag
template <typename T> void tensor_COO<T>::diag(vector<T> &vec) const {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  auto shape = get_shape();
  auto s = *std::min_element(shape.begin(), shape.end());
  assert(s == vec.size());

  for (auto i = decltype(vec.size()){0}; i < vec.size(); i++) {
    vec[i] = 0;
  }

  for (auto i = decltype(vec.size()){0}; i < get_nnz(); i++) {
    bool flag = true;
    for (auto j = decltype(index[i].size()){1}; j < index[i].size(); j++) {
      if (index[i][0] != index[i][j]) {
        flag = false;
      }
    }
    if (flag) {
      vec[index[i][0]] = data()[i];
    }
  }

  logger.func_out();
}
template void
monolish::tensor::tensor_COO<double>::diag(vector<double> &vec) const;
template void
monolish::tensor::tensor_COO<float>::diag(vector<float> &vec) const;

template <typename T>
void tensor_COO<T>::diag(view1D<vector<T>, T> &vec) const {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  auto shape = get_shape();
  auto s = *std::min_element(shape.begin(), shape.end());
  assert(s == vec.size());

  for (auto i = decltype(vec.size()){0}; i < vec.size(); i++) {
    vec[i] = 0;
  }

  for (auto i = decltype(vec.size()){0}; i < get_nnz(); i++) {
    bool flag = true;
    for (auto j = decltype(index[i].size()){1}; j < index[i].size(); j++) {
      if (index[i][0] != index[i][j]) {
        flag = false;
      }
    }
    if (flag) {
      vec[index[i][0]] = data()[i];
    }
  }
  logger.func_out();
}
template void monolish::tensor::tensor_COO<double>::diag(
    view1D<vector<double>, double> &vec) const;
template void monolish::tensor::tensor_COO<float>::diag(
    view1D<vector<float>, float> &vec) const;

template <typename T>
void tensor_COO<T>::diag(view1D<matrix::Dense<T>, T> &vec) const {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  auto shape = get_shape();
  auto s = *std::min_element(shape.begin(), shape.end());
  assert(s == vec.size());

  for (auto i = decltype(vec.size()){0}; i < vec.size(); i++) {
    vec[i] = 0;
  }

  for (auto i = decltype(vec.size()){0}; i < get_nnz(); i++) {
    bool flag = true;
    for (auto j = decltype(index[i].size()){1}; j < index[i].size(); j++) {
      if (index[i][0] != index[i][j]) {
        flag = false;
      }
    }
    if (flag) {
      vec[index[i][0]] = data()[i];
    }
  }

  logger.func_out();
}
template void monolish::tensor::tensor_COO<double>::diag(
    view1D<matrix::Dense<double>, double> &vec) const;
template void monolish::tensor::tensor_COO<float>::diag(
    view1D<matrix::Dense<float>, float> &vec) const;

} // namespace tensor
} // namespace monolish
