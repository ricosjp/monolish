#include "../../../include/monolish/common/monolish_dense.hpp"
#include "../../../include/monolish/common/monolish_logger.hpp"
#include "../../../include/monolish/common/monolish_matrix.hpp"
#include "../../../include/monolish/common/monolish_tensor.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {
namespace matrix {

template <typename T>
void Dense<T>::move(const tensor::tensor_Dense<T> &tensor_dense) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  if (tensor_dense.get_shape().size() != 2) {
    throw std::runtime_error(
        "error cannot move tensor_Dense->Dense when shape.size() != 2");
  }

  this->val_create_flag = false;

  set_row(tensor_dense.get_shape()[0]);
  set_col(tensor_dense.get_shape()[1]);
  this->gpu_status = tensor_dense.get_device_mem_stat();
  this->val = tensor_dense.val;
  this->val_nnz = tensor_dense.get_nnz();
  this->alloc_nnz = tensor_dense.alloc_nnz;

  logger.util_out();
}

template void
Dense<double>::move(const tensor::tensor_Dense<double> &tensor_dense);
template void
Dense<float>::move(const tensor::tensor_Dense<float> &tensor_dense);

template <typename T>
void Dense<T>::move(const tensor::tensor_Dense<T> &tensor_dense, int rowN,
                    int colN) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  if (rowN < 0 && colN < 0) {
    throw std::runtime_error(
        "error cannot move tensor_Dense->Dense when rowN<0 and colN<0");
  }
  if (rowN < 0) {
    rowN = tensor_dense.get_nnz() / colN;
  }
  if (colN < 0) {
    colN = tensor_dense.get_nnz() / rowN;
  }

  if (rowN * colN != tensor_dense.get_nnz()) {
    throw std::runtime_error(
        "error cannot move tensor_Dense->Dense when rowN * colN != get_nnz()");
  }

  this->val_create_flag = false;

  set_row(rowN);
  set_col(colN);
  this->gpu_status = tensor_dense.get_device_mem_stat();
  this->val = tensor_dense.val;
  this->val_nnz = tensor_dense.get_nnz();
  this->alloc_nnz = tensor_dense.alloc_nnz;

  logger.util_out();
}

template void
Dense<double>::move(const tensor::tensor_Dense<double> &tensor_dense, int rowN,
                    int colN);
template void
Dense<float>::move(const tensor::tensor_Dense<float> &tensor_dense, int rowN,
                   int colN);

} // namespace matrix
} // namespace monolish
