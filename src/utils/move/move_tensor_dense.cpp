#include "../../../include/monolish/common/monolish_dense.hpp"
#include "../../../include/monolish/common/monolish_vector.hpp"
#include "../../../include/monolish/common/monolish_logger.hpp"
#include "../../../include/monolish/common/monolish_matrix.hpp"
#include "../../../include/monolish/common/monolish_tensor.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {
namespace tensor {

template <typename T> void tensor_Dense<T>::move(const matrix::Dense<T> &dense){
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  this->shape.resize(2);
  this->shape[0] = dense.get_row();
  this->shape[1] = dense.get_col();

  this->vad_create_flag = false;

  this->gpu_status = dense.get_device_mem_stat();
  this->vad = dense.vad;
  this->vad_nnz = dense.get_nnz();
  this->alloc_nnz = dense.alloc_nnz;

  logger.util_out();
}


template <typename T> void tensor_Dense<T>::move(const vector<T> &vec){
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  this->shape.resize(1);
  this->shape[0] = vec.get_nnz();

  this->vad_create_flag = false;

  this->gpu_status = vec.get_device_mem_stat();
  this->vad = vec.vad;
  this->vad_nnz = vec.get_nnz();
  this->alloc_nnz = vec.alloc_nnz;

  logger.util_out();
}

}
}
