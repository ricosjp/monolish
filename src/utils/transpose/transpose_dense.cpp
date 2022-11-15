#include "../../../include/monolish/common/monolish_dense.hpp"
#include "../../../include/monolish/common/monolish_logger.hpp"
#include "../../../include/monolish/common/monolish_matrix.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {
namespace matrix {

template <typename T> void Dense<T>::transpose() {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);

  size_t M = get_row();
  size_t N = get_col();
  if (M == N) {
    for (size_t i = 0; i < M; i++) {
#pragma omp parallel for
      for (size_t j = i + 1; j < M; j++) {
        std::swap(val[i * M + j], val[j * M + i]);
      }
    }
  } else {
    Dense<T> B(N, M);
#pragma omp parallel for
    for (size_t n = 0; n < M * N; ++n) {
      size_t i = n / get_row();
      size_t j = n % get_row();
      B.val[n] = val[get_col() * j + i];
    }
    internal::vcopy(M * N, B.val.data(), val.data(), false);
    set_row(N);
    set_col(M);
  }

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

#pragma omp parallel for
  for (size_t n = 0; n < get_row() * get_col(); ++n) {
    size_t i = n / get_col();
    size_t j = n % get_col();
    val[n] = B.val[get_row() * j + i];
  }
  logger.util_out();
}
template void Dense<double>::transpose(const Dense<double> &B);
template void Dense<float>::transpose(const Dense<float> &B);

} // namespace matrix
} // namespace monolish
