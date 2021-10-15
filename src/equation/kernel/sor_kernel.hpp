#pragma once

namespace monolish {
namespace {
template <typename T>
void sor_kernel_lower(const monolish::matrix::CRS<T> &A, const vector<T> &D,
                      vector<T> &x, const vector<T> &b) {

  for (int i = 0; i < A.get_row(); i++) {
    auto tmp = b.data()[i];
    for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; j++) {
      if (i > A.col_ind[j]) { // lower
        tmp -= A.val[j] * x.data()[A.col_ind[j]];
      } else {
        break;
      }
    }
    x.data()[i] = tmp * D.data()[i];
  }
}

template <typename T>
void sor_kernel_lower(const monolish::matrix::Dense<T> &A, const vector<T> &D,
                      vector<T> &x, const vector<T> &b) {

  for (int i = 0; i < A.get_row(); i++) {
    auto tmp = b.data()[i];
    for (int j = 0; j < i; j++) {
      tmp -= A.val[i * A.get_col() + j] * x.data()[j];
    }
    x.data()[i] = tmp * D.data()[i];
  }
}
} // namespace
} // namespace monolish
