#include "../../../include/monolish/common/monolish_dense.hpp"
#include "../../../include/monolish/common/monolish_logger.hpp"
#include "../../../include/monolish/common/monolish_matrix.hpp"
#include "../../../include/monolish/common/monolish_tensor.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {

template <typename TYPE, typename T>
void view_Dense<TYPE, T>::move(const view_tensor_Dense<TYPE, T> &view_tensor_dense) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  if (view_tensor_dense.get_shape().size() != 2) {
    throw std::runtime_error(
        "error cannot move tensor_Dense->Dense when shape.size() != 2");
  }

  this->first = view_tensor_dense.get_first();
  this->last = view_tensor_dense.get_last();
  this->rowN = view_tensor_dense.get_shape()[0];
  this->colN = view_tensor_dense.get_shape()[1];
  this->range = view_tensor_dense.get_nnz();
  this->target = view_tensor_dense.get_target();
  this->target_data = view_tensor_dense.data();

  logger.util_out();
}

template void
view_Dense<vector<double>, double>::move(const view_tensor_Dense<vector<double>, double> &view_tensor_dense);
template void
view_Dense<matrix::Dense<double>, double>::move(const view_tensor_Dense<matrix::Dense<double>, double> &view_tensor_dense);
template void
view_Dense<tensor::tensor_Dense<double>, double>::move(const view_tensor_Dense<tensor::tensor_Dense<double>, double> &view_tensor_dense);
template void
view_Dense<vector<float>, float>::move(const view_tensor_Dense<vector<float>, float> &view_tensor_dense);
template void
view_Dense<matrix::Dense<float>, float>::move(const view_tensor_Dense<matrix::Dense<float>, float> &view_tensor_dense);
template void
view_Dense<tensor::tensor_Dense<float>, float>::move(const view_tensor_Dense<tensor::tensor_Dense<float>, float> &view_tensor_dense);

template <typename TYPE, typename T>
void view_Dense<TYPE, T>::move(const view_tensor_Dense<TYPE, T> &view_tensor_dense, int rowN,
                    int colN) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  if (rowN < 0 && colN < 0) {
    throw std::runtime_error(
        "error cannot move tensor_Dense->Dense when rowN<0 and colN<0");
  }
  if (rowN < 0) {
    rowN = view_tensor_dense.get_nnz() / colN;
  }
  if (colN < 0) {
    colN = view_tensor_dense.get_nnz() / rowN;
  }

  if (rowN * colN != view_tensor_dense.get_nnz()) {
    throw std::runtime_error(
        "error cannot move tensor_Dense->Dense when rowN * colN != get_nnz()");
  }

  this->first = view_tensor_dense.get_first();
  this->last = view_tensor_dense.get_last();
  this->rowN = view_tensor_dense.get_shape()[0];
  this->colN = view_tensor_dense.get_shape()[1];
  this->range = view_tensor_dense.get_nnz();
  this->target = view_tensor_dense.get_target();
  this->target_data = view_tensor_dense.data();



  logger.util_out();
}

template void view_Dense<vector<double>, double>::move(const view_tensor_Dense<vector<double>, double> &view_tensor_dense, int rowN, int colN);
template void view_Dense<matrix::Dense<double>, double>::move(const view_tensor_Dense<matrix::Dense<double>, double> &view_tensor_dense, int rowN, int colN);
template void view_Dense<tensor::tensor_Dense<double>, double>::move(const view_tensor_Dense<tensor::tensor_Dense<double>, double> &view_tensor_dense, int rowN, int colN);
template void view_Dense<vector<float>, float>::move(const view_tensor_Dense<vector<float>, float> &view_tensor_dense, int rowN, int colN);
template void view_Dense<matrix::Dense<float>, float>::move(const view_tensor_Dense<matrix::Dense<float>, float> &view_tensor_dense, int rowN, int colN);
template void view_Dense<tensor::tensor_Dense<float>, float>::move(const view_tensor_Dense<tensor::tensor_Dense<float>, float> &view_tensor_dense, int rowN, int colN);

} // namespace monolish
