#include "../../../include/common/monolish_dense.hpp"
#include "../../../include/common/monolish_logger.hpp"
#include "../../../include/common/monolish_matrix.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {
namespace matrix {

template <typename T> void COO<T>::create_hash() {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  structure_hash = internal::vhash(row_index.size(), row_index.data(),
                                   get_row(), get_device_mem_stat());
  structure_hash = internal::vhash(col_index.size(), col_index.data(),
                                   structure_hash, get_device_mem_stat());

  logger.util_out();
}
template void COO<double>::create_hash();
template void COO<float>::create_hash();

} // namespace matrix
} // namespace monolish
