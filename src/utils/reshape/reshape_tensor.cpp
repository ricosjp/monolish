#include "../../../include/monolish/common/monolish_tensor.hpp"
#include "../../../include/monolish/common/monolish_logger.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {
namespace tensor {

template <typename T>
void tensor_Dense<T>::reshape(const std::vector<size_t>& shape) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  int minus = 0;
  size_t N = 1;
  for(auto n : shape){
    if(n<0){
      minus++;
    }else{
      N *= n;
    }
  }
  if(minus >= 2){
    throw std::runtime_error(
        "negative value of shape should be 0 or 1");
  }

  this->shape = shape;

  if(minus==1){
    std::size_t M = 1;
    for(size_t index=0; index<this->shape.size(); ++index){
      if(this->shape[index] < 0){
        this->shape[index] = get_nnz()/N;
      }
      M *= this->shape[index];
    }
    N = M;
  }

  if (N != get_nnz()) {
    throw std::runtime_error(
        "error size should be unchanged when matrix is reshaped");
  }

  logger.util_out();
}
template void tensor_Dense<double>::reshape(const std::vector<size_t>& shape);
template void tensor_Dense<float>::reshape(const std::vector<size_t>& shape);

} // namespace matrix
} // namespace monolish
