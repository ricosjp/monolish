#include "../../../include/monolish/common/monolish_dense.hpp"
#include "../../../include/monolish/common/monolish_logger.hpp"
#include "../../../include/monolish/common/monolish_matrix.hpp"
#include "../../../include/monolish/common/monolish_tensor.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {
namespace matrix {

template <typename T> void Dense<T>::move(const tensor::tensor_Dense<T> &tensor_dense){
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  if(shape.size() != 2){
    throw std::runtime_error(
        "error cannot move tensor_Dense->Dense when shape.size() != 2");
  }

  this->vad_create_flag = false;

  this->gpu_status = tensor_dense.get_device_mem_stat();
  this->vad = tensor_dense.vad;
  this->vad_nnz = tensor_dense.get_nnz();
  this->alloc_nnz = tensor_dense.alloc_nnz;

  logger.util_out();
}

}
}
