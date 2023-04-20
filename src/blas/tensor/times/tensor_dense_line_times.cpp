#include "../../../../include/monolish_blas.hpp"
#include "../../../internal/monolish_internal.hpp"

namespace monolish {

namespace {

// vector-tensor_row times //
template <typename T, typename VEC>
void times_row_core(const tensor::tensor_Dense<T> &A, const size_t num,
                    const VEC &x, tensor::tensor_Dense<T> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  assert(util::is_same_structure(A, C));
  assert(util::is_same_device_mem_stat(A, x, C));

  matrix::Dense<T> Amat;
  Amat.move(A, -1, A.get_shape()[A.get_shape().size() - 1]);
  assert(Amat.get_col() == x.size());
  assert(Amat.get_row() >= num);

  const auto *Ad = A.data();
  const auto n = Amat.get_col();
  auto *Cd = C.data();

  const auto *xd = x.data();
  const auto xoffset = x.get_offset();

  if (A.get_device_mem_stat() == true) {
#if MONOLISH_USE_NVIDIA_GPU
#pragma omp target teams distribute parallel for
    for (auto j = decltype(n){0}; j < n; j++) {
      Cd[num * n + j] = Ad[num * n + j] * xd[j + xoffset];
    }
#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  } else {
#pragma omp parallel for
    for (auto j = decltype(n){0}; j < n; j++) {
      Cd[num * n + j] = Ad[num * n + j] * xd[j + xoffset];
    }
  }

  logger.func_out();
}

// vector-tensor_col times //
template <typename T, typename VEC>
void times_col_core(const tensor::tensor_Dense<T> &A, const size_t num,
                    const VEC &x, tensor::tensor_Dense<T> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  assert(util::is_same_structure(A, C));
  assert(util::is_same_device_mem_stat(A, x, C));

  matrix::Dense<T> Amat;
  Amat.move(A, A.get_shape()[0], -1);
  assert(Amat.get_row() == x.size());
  assert(Amat.get_col() >= num);

  const auto *Ad = A.data();
  const auto m = Amat.get_row();
  const auto n = Amat.get_col();
  auto *Cd = C.data();

  const auto *xd = x.data();
  const auto xoffset = x.get_offset();

  if (A.get_device_mem_stat() == true) {
#if MONOLISH_USE_NVIDIA_GPU
#pragma omp target teams distribute parallel for
    for (auto i = decltype(m){0}; i < m; i++) {
      Cd[i * n + num] = Ad[i * n + num] * xd[i + xoffset];
    }
#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  } else {
#pragma omp parallel for
    for (auto i = decltype(m){0}; i < m; i++) {
      Cd[i * n + num] = Ad[i * n + num] * xd[i + xoffset];
    }
  }

  logger.func_out();
}
} // namespace

namespace blas {

// vector-tensor_row times line //
void times_row(const tensor::tensor_Dense<double> &A, const size_t num,
               const vector<double> &x, tensor::tensor_Dense<double> &C) {
  times_row_core(A, num, x, C);
}
void times_row(const tensor::tensor_Dense<double> &A, const size_t num,
               const view1D<vector<double>, double> &x,
               tensor::tensor_Dense<double> &C) {
  times_row_core(A, num, x, C);
}
void times_row(const tensor::tensor_Dense<double> &A, const size_t num,
               const view1D<tensor::tensor_Dense<double>, double> &x,
               tensor::tensor_Dense<double> &C) {
  times_row_core(A, num, x, C);
}
void times_row(const tensor::tensor_Dense<float> &A, const size_t num,
               const vector<float> &x, tensor::tensor_Dense<float> &C) {
  times_row_core(A, num, x, C);
}
void times_row(const tensor::tensor_Dense<float> &A, const size_t num,
               const view1D<vector<float>, float> &x,
               tensor::tensor_Dense<float> &C) {
  times_row_core(A, num, x, C);
}
void times_row(const tensor::tensor_Dense<float> &A, const size_t num,
               const view1D<tensor::tensor_Dense<float>, float> &x,
               tensor::tensor_Dense<float> &C) {
  times_row_core(A, num, x, C);
}

// vector-tensor_col times line //
void times_col(const tensor::tensor_Dense<double> &A, const size_t num,
               const vector<double> &x, tensor::tensor_Dense<double> &C) {
  times_col_core(A, num, x, C);
}
void times_col(const tensor::tensor_Dense<double> &A, const size_t num,
               const view1D<vector<double>, double> &x,
               tensor::tensor_Dense<double> &C) {
  times_col_core(A, num, x, C);
}
void times_col(const tensor::tensor_Dense<double> &A, const size_t num,
               const view1D<tensor::tensor_Dense<double>, double> &x,
               tensor::tensor_Dense<double> &C) {
  times_col_core(A, num, x, C);
}
void times_col(const tensor::tensor_Dense<float> &A, const size_t num,
               const vector<float> &x, tensor::tensor_Dense<float> &C) {
  times_col_core(A, num, x, C);
}
void times_col(const tensor::tensor_Dense<float> &A, const size_t num,
               const view1D<vector<float>, float> &x,
               tensor::tensor_Dense<float> &C) {
  times_col_core(A, num, x, C);
}
void times_col(const tensor::tensor_Dense<float> &A, const size_t num,
               const view1D<tensor::tensor_Dense<float>, float> &x,
               tensor::tensor_Dense<float> &C) {
  times_col_core(A, num, x, C);
}

} // namespace blas
} // namespace monolish
