#include "../../../include/common/monolish_dense.hpp"
#include "../../../include/common/monolish_logger.hpp"
#include "../../../include/common/monolish_matrix.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {
namespace matrix {

template <typename T> void COO<T>::operator=(const matrix::COO<T> &mat) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  // err
  if (get_row() != mat.get_row()) {
    throw std::runtime_error("error A.row != C.row");
  }
  if (get_col() != mat.get_col()) {
    throw std::runtime_error("error A.col != C.col");
  }
  if (get_nnz() != mat.get_nnz()) {
    throw std::runtime_error("error A.nnz != C.nnz");
  }
  if (get_device_mem_stat() != mat.get_device_mem_stat()) {
    throw std::runtime_error("error get_device_mem_stat() is not same");
  }

  // value copy
  internal::vcopy(get_nnz(), val.data(), mat.val.data(), get_device_mem_stat());

  logger.util_out();
}

}
}
