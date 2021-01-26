#include "../../../include/common/monolish_dense.hpp"
#include "../../../include/common/monolish_logger.hpp"
#include "../../../include/common/monolish_matrix.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {
namespace util {

template <typename T>
bool is_same_structure(const matrix::LinearOperator<T> &A,
                       const matrix::LinearOperator<T> &B) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  bool ans = false;

  if (A.get_row() == B.get_row() && A.get_col() == B.get_col()) {
    logger.util_out();
    ans = true;
  }

  logger.util_out();
  return ans;
}

template bool is_same_structure(const matrix::LinearOperator<double> &A,
                                const matrix::LinearOperator<double> &B);
template bool is_same_structure(const matrix::LinearOperator<float> &A,
                                const matrix::LinearOperator<float> &B);

template <typename T>
bool is_same_size(const matrix::LinearOperator<T> &A,
                  const matrix::LinearOperator<T> &B) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  bool ans = true;

  if (A.get_row() != B.get_row() && A.get_col() != B.get_col()) {
    logger.util_out();
    ans = false;
  }

  logger.util_out();
  return ans;
}

template bool is_same_size(const matrix::LinearOperator<double> &A,
                           const matrix::LinearOperator<double> &B);
template bool is_same_size(const matrix::LinearOperator<float> &A,
                           const matrix::LinearOperator<float> &B);

} // namespace util
} // namespace monolish
