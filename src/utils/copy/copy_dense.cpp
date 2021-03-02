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

template <typename T>
void Dense<T>::set_ptr(const size_t M, const size_t N, const std::vector<T> &value) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  val = value;

  rowN = M;
  colN = N;
  nnz = M*N;
  logger.util_out();
}
template void Dense<double>::set_ptr(const size_t M, const size_t N,
                                   const std::vector<double> &value);
template void Dense<float>::set_ptr(const size_t M, const size_t N,
                                  const std::vector<float> &value);

} // namespace matrix
} // namespace monolish
