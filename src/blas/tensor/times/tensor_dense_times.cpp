#include "../../../../include/monolish_blas.hpp"
#include "../../../internal/monolish_internal.hpp"

namespace monolish {

namespace {

// scalar-tensor times //
template <typename T>
void times_core(const T alpha, const tensor::tensor_Dense<T> &A,
                tensor::tensor_Dense<T> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  assert(util::is_same_structure(A, C));
  assert(util::is_same_device_mem_stat(A, C));

  internal::vmul(A.get_nnz(), A.data(), alpha, C.data(),
                 A.get_device_mem_stat());

  logger.func_out();
}

// vector-tensor_row times all //
template <typename T, typename VEC>
void times_row_core(const tensor::tensor_Dense<T> &A, const VEC &x,
                    tensor::tensor_Dense<T> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  assert(util::is_same_structure(A, C));
  assert(util::is_same_device_mem_stat(A, x, C));

  matrix::Dense<T> Amat;
  Amat.move(A, -1, A.get_shape()[A.get_shape().size() - 1]);
  assert(Amat.get_col() == x.size());

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
      for (auto j = decltype(n){0}; j < n; j++) {
        Cd[i * n + j] = Ad[i * n + j] * xd[j + xoffset];
      }
    }
#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  } else {
#pragma omp parallel for
    for (auto i = decltype(m){0}; i < m; i++) {
      for (auto j = decltype(n){0}; j < n; j++) {
        Cd[i * n + j] = Ad[i * n + j] * xd[j + xoffset];
      }
    }
  }

  logger.func_out();
}

// vector-tensor_col times all //
template <typename T, typename VEC>
void times_col_core(const tensor::tensor_Dense<T> &A, const VEC &x,
                    tensor::tensor_Dense<T> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  assert(util::is_same_structure(A, C));
  assert(util::is_same_device_mem_stat(A, x, C));

  matrix::Dense<T> Amat;
  Amat.move(A, A.get_shape()[0], -1);
  assert(Amat.get_row() == x.size());

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
      for (auto j = decltype(n){0}; j < n; j++) {
        Cd[i * n + j] = Ad[i * n + j] * xd[i + xoffset];
      }
    }
#else
    throw std::runtime_error(
        "error USE_GPU is false, but get_device_mem_stat() == true");
#endif
  } else {
#pragma omp parallel for
    for (auto i = decltype(m){0}; i < m; i++) {
      for (auto j = decltype(n){0}; j < n; j++) {
        Cd[i * n + j] = Ad[i * n + j] * xd[i + xoffset];
      }
    }
  }

  logger.func_out();
}
} // namespace

namespace blas {

// scalar-tensor times //
void times(const double alpha, const tensor::tensor_Dense<double> &A,
           tensor::tensor_Dense<double> &C) {
  times_core(alpha, A, C);
}

void times(const float alpha, const tensor::tensor_Dense<float> &A,
           tensor::tensor_Dense<float> &C) {
  times_core(alpha, A, C);
}

// vector-tensor_row times all //
void times_row(const tensor::tensor_Dense<double> &A, const vector<double> &x,
               tensor::tensor_Dense<double> &C) {
  times_row_core(A, x, C);
}
void times_row(const tensor::tensor_Dense<double> &A,
               const view1D<vector<double>, double> &x,
               tensor::tensor_Dense<double> &C) {
  times_row_core(A, x, C);
}
void times_row(const tensor::tensor_Dense<double> &A,
               const view1D<tensor::tensor_Dense<double>, double> &x,
               tensor::tensor_Dense<double> &C) {
  times_row_core(A, x, C);
}
void times_row(const tensor::tensor_Dense<float> &A, const vector<float> &x,
               tensor::tensor_Dense<float> &C) {
  times_row_core(A, x, C);
}
void times_row(const tensor::tensor_Dense<float> &A,
               const view1D<vector<float>, float> &x,
               tensor::tensor_Dense<float> &C) {
  times_row_core(A, x, C);
}
void times_row(const tensor::tensor_Dense<float> &A,
               const view1D<tensor::tensor_Dense<float>, float> &x,
               tensor::tensor_Dense<float> &C) {
  times_row_core(A, x, C);
}

// vector-tensor_col times all //
void times_col(const tensor::tensor_Dense<double> &A, const vector<double> &x,
               tensor::tensor_Dense<double> &C) {
  times_col_core(A, x, C);
}
void times_col(const tensor::tensor_Dense<double> &A,
               const view1D<vector<double>, double> &x,
               tensor::tensor_Dense<double> &C) {
  times_col_core(A, x, C);
}
void times_col(const tensor::tensor_Dense<double> &A,
               const view1D<tensor::tensor_Dense<double>, double> &x,
               tensor::tensor_Dense<double> &C) {
  times_col_core(A, x, C);
}
void times_col(const tensor::tensor_Dense<float> &A, const vector<float> &x,
               tensor::tensor_Dense<float> &C) {
  times_col_core(A, x, C);
}
void times_col(const tensor::tensor_Dense<float> &A,
               const view1D<vector<float>, float> &x,
               tensor::tensor_Dense<float> &C) {
  times_col_core(A, x, C);
}
void times_col(const tensor::tensor_Dense<float> &A,
               const view1D<tensor::tensor_Dense<float>, float> &x,
               tensor::tensor_Dense<float> &C) {
  times_col_core(A, x, C);
}

} // namespace blas
} // namespace monolish
