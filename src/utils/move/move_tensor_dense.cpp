#include "../../../include/monolish/common/monolish_dense.hpp"
#include "../../../include/monolish/common/monolish_logger.hpp"
#include "../../../include/monolish/common/monolish_matrix.hpp"
#include "../../../include/monolish/common/monolish_tensor.hpp"
#include "../../../include/monolish/common/monolish_vector.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {
namespace tensor {

template <typename T>
void tensor_Dense<T>::move(const matrix::Dense<T> &dense) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  this->shape.resize(2);
  this->shape[0] = dense.get_row();
  this->shape[1] = dense.get_col();

  this->val_create_flag = false;

  this->gpu_status = dense.get_gpu_status();
  this->val = dense.val;
  this->val_nnz = dense.get_nnz();
  this->alloc_nnz = dense.alloc_nnz;
  set_first(dense.get_first());

  logger.util_out();
}

template void tensor_Dense<double>::move(const matrix::Dense<double> &dense);
template void tensor_Dense<float>::move(const matrix::Dense<float> &dense);

template <typename T> void tensor_Dense<T>::move(const vector<T> &vec) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  this->shape.resize(1);
  this->shape[0] = vec.get_nnz();

  this->val_create_flag = false;

  this->gpu_status = vec.get_gpu_status();
  this->val = vec.val;
  this->val_nnz = vec.get_nnz();
  this->alloc_nnz = vec.alloc_nnz;
  set_first(vec.get_first());

  logger.util_out();
}

template void tensor_Dense<double>::move(const vector<double> &dense);
template void tensor_Dense<float>::move(const vector<float> &dense);

} // namespace tensor
} // namespace monolish
