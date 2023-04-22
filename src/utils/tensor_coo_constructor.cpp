#include "../../include/monolish/common/monolish_dense.hpp"
#include "../../include/monolish/common/monolish_logger.hpp"
#include "../../include/monolish/common/monolish_matrix.hpp"
#include "../../include/monolish/common/monolish_tensor_dense.hpp"
#include "../../include/monolish/common/monolish_vector.hpp"
#include "../internal/monolish_internal.hpp"

namespace monolish {
namespace tensor {

template <typename T>
tensor_COO<T>::tensor_COO(const std::vector<size_t> &shape,
                          const std::vector<std::vector<size_t>> &index,
                          const T *value) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  this->shape = shape;
  this->index = index;

  val_create_flag = true;
  resize(index.size());
  std::copy(value, value + get_nnz(), data());
  logger.util_out();
}

template tensor_COO<double>::tensor_COO(
    const std::vector<size_t> &shape,
    const std::vector<std::vector<size_t>> &index, const double *value);
template tensor_COO<float>::tensor_COO(
    const std::vector<size_t> &shape,
    const std::vector<std::vector<size_t>> &index, const float *value);

template <typename T> tensor_COO<T>::tensor_COO(const tensor_COO<T> &coo) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  this->shape = coo.shape;
  this->index = coo.index;
  gpu_status = false;

  val_create_flag = true;
  resize(coo.get_nnz());
  std::copy(coo.data(), coo.data() + coo.get_nnz(), data());
  logger.util_out();
}

template tensor_COO<double>::tensor_COO(const tensor_COO<double> &coo);
template tensor_COO<float>::tensor_COO(const tensor_COO<float> &coo);

template <typename T>
tensor_COO<T>::tensor_COO(const tensor_COO<T> &coo, T value) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  this->shape = coo.shape;
  this->index = coo.index;
  gpu_status = false;

  val_create_flag = true;
  resize(coo.get_nnz());
  internal::vbroadcast(coo.get_nnz(), value, data(), false);
  logger.util_out();
}

template tensor_COO<double>::tensor_COO(const tensor_COO<double> &coo,
                                        double value);
template tensor_COO<float>::tensor_COO(const tensor_COO<float> &coo,
                                       float value);

} // namespace tensor
} // namespace monolish
