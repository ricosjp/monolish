#include "../../include/monolish/common/monolish_dense.hpp"
#include "../../include/monolish/common/monolish_logger.hpp"
#include "../../include/monolish/common/monolish_matrix.hpp"
#include "../../include/monolish/common/monolish_tensor.hpp"
#include "../internal/monolish_internal.hpp"

namespace monolish {
namespace matrix {

template <typename T> void CRS<T>::compute_hash() {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  structure_hash = internal::vhash(row_ptr.size(), row_ptr.data(), get_row(),
                                   get_device_mem_stat());
  structure_hash = internal::vhash(col_ind.size(), col_ind.data(),
                                   structure_hash, get_device_mem_stat());

  logger.util_out();
}
template void CRS<double>::compute_hash();
template void CRS<float>::compute_hash();

} // namespace matrix

namespace tensor {

template <typename T> void tensor_CRS<T>::compute_hash() {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  structure_hash = 0;

  for (size_t d = 0; d < row_ptrs.size(); ++d) {
    structure_hash = internal::vhash(row_ptrs[d].size(), row_ptrs[d].data(),
                                     shape[0] | structure_hash, get_device_mem_stat());
    structure_hash = internal::vhash(col_inds[d].size(), col_inds[d].data(),
                                     structure_hash, get_device_mem_stat());
  }

  logger.util_out();
}
template void tensor_CRS<double>::compute_hash();
template void tensor_CRS<float>::compute_hash();

} // namespace tensor

} // namespace monolish
