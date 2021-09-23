#include "../../../include/common/monolish_dense.hpp"
#include "../../../include/common/monolish_logger.hpp"
#include "../../../include/common/monolish_matrix.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {
namespace matrix {

template <typename T> void &Dense<T>::transpose() {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  size_t M = get_row();
  size_t N = get_col();

  for (int i = 0; i < M; i++){
#pragma omp parallel for
      for (int j = i+1; j < N; j++) {
          swap(A[i*M+j], A[j*N+i]);
      }
  }
  set_row(col);
  set_col(row);
  
  logger.util_out();
}
template void Dense<double>::transpose();
template void Dense<float>::transpose();

template <typename T> void Dense<T>::transpose(const Dense<T> &B) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  set_row(B.get_col());
  set_col(B.get_row());
  val.resize(B.get_row() * B.get_col());

  for (size_t i = 0; i < get_row(); ++i) {
    for (size_t j = 0; j < get_col(); ++j) {
      val[i * get_col() + j] = B.val[j * get_row() + i];
    }
  }
  logger.util_out();
}
template void Dense<double>::transpose(const Dense<double> &B);
template void Dense<float>::transpose(const Dense<float> &B);

} // namespace matrix
} // namespace monolish
