#include "../../include/monolish/common/monolish_dense.hpp"
#include "../../include/monolish/common/monolish_logger.hpp"
#include "../../include/monolish/common/monolish_matrix.hpp"
#include "../../include/monolish/common/monolish_tensor_coo.hpp"
#include "../../include/monolish/common/monolish_tensor_dense.hpp"
#include "../../include/monolish/common/monolish_vector.hpp"
#include "../internal/monolish_internal.hpp"

namespace monolish {
namespace tensor {

template <typename T>
tensor_CRS<T>::tensor_CRS(const std::vector<size_t> &shape,
                          const std::vector<std::vector<int>> &row_ptrs,
                          const std::vector<std::vector<int>> &col_inds,
                          const T *value) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  this->shape = shape;
  this->row_ptrs = row_ptrs;
  this->col_inds = col_inds;

  val_create_flag = true;
  size_t num = 0;
  for (size_t d = 0; d < col_inds.size(); ++d) {
    num += col_inds[d].size();
  }
  resize(num);
  std::copy(value, value + get_nnz(), data());
  logger.util_out();
}

template tensor_CRS<double>::tensor_CRS(
    const std::vector<size_t> &shape,
    const std::vector<std::vector<int>> &row_ptrs,
    const std::vector<std::vector<int>> &col_inds, const double *value);
template tensor_CRS<float>::tensor_CRS(
    const std::vector<size_t> &shape,
    const std::vector<std::vector<int>> &row_ptrs,
    const std::vector<std::vector<int>> &col_inds, const float *value);

template <typename T>
tensor_CRS<T>::tensor_CRS(const tensor_CRS<T> &crs, T value) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  this->shape = crs.shape;
  this->row_ptrs = crs.row_ptrs;
  this->col_inds = crs.col_inds;
  gpu_status = false;

  val_create_flag = true;
  resize(crs.get_nnz());
  internal::vbroadcast(crs.get_nnz(), value, data(), false);
  logger.util_out();
}

template tensor_CRS<double>::tensor_CRS(const tensor_CRS<double> &coo,
                                        double value);
template tensor_CRS<float>::tensor_CRS(const tensor_CRS<float> &coo,
                                       float value);

} // namespace tensor
} // namespace monolish
