#include "../../../include/common/monolish_common.hpp"
#include "../../../include/common/monolish_dense.hpp"
#include "../../../include/common/monolish_logger.hpp"
#include "../../../include/common/monolish_matrix.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {
namespace matrix {

template <typename T> void Dense<T>::operator=(const Dense<T> &mat) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  // err
  assert(monolish::util::is_same_size(*this, mat));
  assert(monolish::util::is_same_device_mem_stat(*this, mat));

  // gpu copy
  if (mat.get_device_mem_stat()) {
    internal::vcopy(get_nnz(), mat.val.data(), val.data(), true);
  } else {
    internal::vcopy(get_nnz(), mat.val.data(), val.data(), false);
  }

  logger.util_out();
}

template void Dense<double>::operator=(const Dense<double> &mat);
template void Dense<float>::operator=(const Dense<float> &mat);

} // namespace matrix
} // namespace monolish
