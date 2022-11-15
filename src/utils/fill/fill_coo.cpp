#include "../../../include/monolish/common/monolish_dense.hpp"
#include "../../../include/monolish/common/monolish_logger.hpp"
#include "../../../include/monolish/common/monolish_matrix.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {
namespace matrix {

template <typename T> void COO<T>::fill(T value) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
#pragma omp parallel for
  for (auto i = decltype(get_nnz()){0}; i < get_nnz(); i++) {
    val[i] = value;
  }
  logger.util_out();
}
template void COO<double>::fill(double value);
template void COO<float>::fill(float value);

} // namespace matrix
} // namespace monolish
