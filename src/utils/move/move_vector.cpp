#include "../../../include/monolish/common/monolish_dense.hpp"
#include "../../../include/monolish/common/monolish_logger.hpp"
#include "../../../include/monolish/common/monolish_matrix.hpp"
#include "../../../include/monolish/common/monolish_tensor.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {

template <typename T>
void vector<T>::move(const tensor::tensor_Dense<T> &tensor_dense) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  if (tensor_dense.get_shape().size() != 1) {
    throw std::runtime_error(
        "error cannot move tensor_Dense->vector when shape.size() != 1");
  }

  this->val_create_flag = false;

  this->gpu_status = tensor_dense.get_gpu_status();
  this->val = tensor_dense.val;
  this->val_nnz = tensor_dense.get_nnz();
  this->alloc_nnz = tensor_dense.alloc_nnz;
  set_first(tensor_dense.get_first());

  logger.util_out();
}

template void
vector<double>::move(const tensor::tensor_Dense<double> &tensor_dense);
template void
vector<float>::move(const tensor::tensor_Dense<float> &tensor_dense);

template <typename T>
void vector<T>::move(const view_tensor_Dense<vector<T>, T> &tensor_dense) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  if (tensor_dense.get_shape().size() != 1) {
    throw std::runtime_error(
        "error cannot move tensor_Dense->vector when shape.size() != 1");
  }

  this->val_create_flag = false;

  this->gpu_status = tensor_dense.get_gpu_status();
  this->val = tensor_dense.get_val();
  this->val_nnz = tensor_dense.get_nnz();
  this->alloc_nnz = tensor_dense.get_alloc_nnz();
  set_first(tensor_dense.get_first());

  logger.util_out();
}

template void vector<double>::move(
    const view_tensor_Dense<vector<double>, double> &tensor_dense);
template void vector<float>::move(
    const view_tensor_Dense<vector<float>, float> &tensor_dense);

template <typename T>
void vector<T>::move(
    const view_tensor_Dense<matrix::Dense<T>, T> &tensor_dense) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  if (tensor_dense.get_shape().size() != 1) {
    throw std::runtime_error(
        "error cannot move tensor_Dense->vector when shape.size() != 1");
  }

  this->val_create_flag = false;

  this->gpu_status = tensor_dense.get_gpu_status();
  this->val = tensor_dense.get_val();
  this->val_nnz = tensor_dense.get_nnz();
  this->alloc_nnz = tensor_dense.get_alloc_nnz();
  set_first(tensor_dense.get_first());

  logger.util_out();
}

template void vector<double>::move(
    const view_tensor_Dense<matrix::Dense<double>, double> &tensor_dense);
template void vector<float>::move(
    const view_tensor_Dense<matrix::Dense<float>, float> &tensor_dense);

template <typename T>
void vector<T>::move(
    const view_tensor_Dense<tensor::tensor_Dense<T>, T> &tensor_dense) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  if (tensor_dense.get_shape().size() != 1) {
    throw std::runtime_error(
        "error cannot move tensor_Dense->vector when shape.size() != 1");
  }

  this->val_create_flag = false;

  this->gpu_status = tensor_dense.get_gpu_status();
  this->val = tensor_dense.get_val();
  this->val_nnz = tensor_dense.get_nnz();
  this->alloc_nnz = tensor_dense.get_alloc_nnz();
  set_first(tensor_dense.get_first());

  logger.util_out();
}

template void vector<double>::move(
    const view_tensor_Dense<tensor::tensor_Dense<double>, double>
        &tensor_dense);
template void vector<float>::move(
    const view_tensor_Dense<tensor::tensor_Dense<float>, float> &tensor_dense);

template <typename T>
void vector<T>::move(const tensor::tensor_Dense<T> &tensor_dense, int nnz) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  if (nnz < 0) {
    nnz = tensor_dense.get_nnz();
  }
  if (nnz != tensor_dense.get_nnz()) {
    throw std::runtime_error(
        "error cannot move tensor_Dense->vector when nnz != get_nnz()");
  }

  this->val_create_flag = false;

  this->gpu_status = tensor_dense.get_gpu_status();
  this->val = tensor_dense.val;
  this->val_nnz = tensor_dense.get_nnz();
  this->alloc_nnz = tensor_dense.alloc_nnz;
  set_first(tensor_dense.get_first());

  logger.util_out();
}

template void
vector<double>::move(const tensor::tensor_Dense<double> &tensor_dense, int nnz);
template void
vector<float>::move(const tensor::tensor_Dense<float> &tensor_dense, int nnz);

template <typename T>
void vector<T>::move(const view_tensor_Dense<vector<T>, T> &tensor_dense,
                     int nnz) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  if (nnz < 0) {
    nnz = tensor_dense.get_nnz();
  }
  if (nnz != tensor_dense.get_nnz()) {
    throw std::runtime_error(
        "error cannot move tensor_Dense->vector when nnz != get_nnz()");
  }

  this->val_create_flag = false;

  this->gpu_status = tensor_dense.get_gpu_status();
  this->val = tensor_dense.get_val();
  this->val_nnz = tensor_dense.get_nnz();
  this->alloc_nnz = tensor_dense.get_alloc_nnz();
  set_first(tensor_dense.get_first());

  logger.util_out();
}

template void vector<double>::move(
    const view_tensor_Dense<vector<double>, double> &tensor_dense, int nnz);
template void
vector<float>::move(const view_tensor_Dense<vector<float>, float> &tensor_dense,
                    int nnz);

template <typename T>
void vector<T>::move(const view_tensor_Dense<matrix::Dense<T>, T> &tensor_dense,
                     int nnz) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  if (nnz < 0) {
    nnz = tensor_dense.get_nnz();
  }
  if (nnz != tensor_dense.get_nnz()) {
    throw std::runtime_error(
        "error cannot move tensor_Dense->vector when nnz != get_nnz()");
  }

  this->val_create_flag = false;

  this->gpu_status = tensor_dense.get_gpu_status();
  this->val = tensor_dense.get_val();
  this->val_nnz = tensor_dense.get_nnz();
  this->alloc_nnz = tensor_dense.get_alloc_nnz();
  set_first(tensor_dense.get_first());

  logger.util_out();
}

template void vector<double>::move(
    const view_tensor_Dense<matrix::Dense<double>, double> &tensor_dense,
    int nnz);
template void vector<float>::move(
    const view_tensor_Dense<matrix::Dense<float>, float> &tensor_dense,
    int nnz);

template <typename T>
void vector<T>::move(
    const view_tensor_Dense<tensor::tensor_Dense<T>, T> &tensor_dense,
    int nnz) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  if (nnz < 0) {
    nnz = tensor_dense.get_nnz();
  }
  if (nnz != tensor_dense.get_nnz()) {
    throw std::runtime_error(
        "error cannot move tensor_Dense->vector when nnz != get_nnz()");
  }

  this->val_create_flag = false;

  this->gpu_status = tensor_dense.get_gpu_status();
  this->val = tensor_dense.get_val();
  this->val_nnz = tensor_dense.get_nnz();
  this->alloc_nnz = tensor_dense.get_alloc_nnz();
  set_first(tensor_dense.get_first());

  logger.util_out();
}

template void vector<double>::move(
    const view_tensor_Dense<tensor::tensor_Dense<double>, double> &tensor_dense,
    int nnz);
template void vector<float>::move(
    const view_tensor_Dense<tensor::tensor_Dense<float>, float> &tensor_dense,
    int nnz);

} // namespace monolish
