#include "../../../include/common/monolish_dense.hpp"
#include "../../../include/common/monolish_logger.hpp"
#include "../../../include/common/monolish_matrix.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {
namespace matrix {

template <typename T> void CRS<T>::operator=(const CRS<T> &mat) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  // err
  assert(monolish::util::is_same_size(*this, mat));
  assert(monolish::util::is_same_structure(*this, mat));
  assert(monolish::util::is_same_device_mem_stat(*this, mat));

  if (mat.get_device_mem_stat() == true) {
#if MONOLISH_USE_GPU
    internal::vcopy(mat.row_ptr.size(), mat.row_ptr.data(), row_ptr.data(),
                    true);
    internal::vcopy(mat.col_ind.size(), mat.col_ind.data(), col_ind.data(),
                    true);
    internal::vcopy(mat.val.size(), mat.val.data(), val.data(), true);
#endif
  } else {
    internal::vcopy(mat.row_ptr.size(), mat.row_ptr.data(), row_ptr.data(),
                    false);
    internal::vcopy(mat.col_ind.size(), mat.col_ind.data(), col_ind.data(),
                    false);
    internal::vcopy(mat.val.size(), mat.val.data(), val.data(), false);
  }

  logger.util_out();
}
template void CRS<double>::operator=(const CRS<double> &mat);
template void CRS<float>::operator=(const CRS<float> &mat);

} // namespace matrix
} // namespace monolish
