#include "../../../include/monolish/common/monolish_dense.hpp"
#include "../../../include/monolish/common/monolish_logger.hpp"
#include "../../../include/monolish/common/monolish_matrix.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {
namespace matrix {

template <typename T>
void Dense<T>::reshape(const size_t new_row, const size_t new_col) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  size_t M = get_row();
  size_t N = get_col();

  if (M * N != new_row * new_col) {
    throw std::runtime_error(
        "error size should be unchanged when matrix is reshaped");
  }

  set_row(new_row);
  set_col(new_col);

  logger.util_out();
}
template void Dense<double>::reshape(const size_t new_row,
                                     const size_t new_col);
template void Dense<float>::reshape(const size_t new_row, const size_t new_col);

} // namespace matrix
} // namespace monolish
