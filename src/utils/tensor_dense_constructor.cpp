#include "../../include/monolish/common/monolish_dense.hpp"
#include "../../include/monolish/common/monolish_logger.hpp"
#include "../../include/monolish/common/monolish_matrix.hpp"
#include "../../include/monolish/common/monolish_tensor_dense.hpp"
#include "../../include/monolish/common/monolish_vector.hpp"
#include "../internal/monolish_internal.hpp"

namespace monolish {
namespace tensor {

template <typename T>
tensor_Dense<T>::tensor_Dense(const std::vector<size_t> &shape) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  this->shape = shape;

  vad_create_flag = true;
  resize(shape);
  logger.util_out();
}

template <typename T>
tensor_Dense<T>::tensor_Dense(const std::vector<size_t> &shape,
                              const T *value) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  this->shape = shape;

  vad_create_flag = true;
  resize(shape);
  std::copy(value, value + get_nnz(), data());
  logger.util_out();
}

} // namespace tensor
} // namespace monolish
