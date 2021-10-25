#include "../../../include/monolish_blas.hpp"
#include "../../internal/monolish_internal.hpp"

namespace monolish {
namespace matrix {

// diag
template <typename T> void COO<T>::diag(vector<T> &vec) const {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  auto s = get_row() > get_col() ? get_col() : get_row();
  assert(s == vec.size());

  for (auto i = decltype(vec.size()){0}; i < vec.size(); i++) {
    vec[i] = 0;
  }

  for (auto i = decltype(vec.size()){0}; i < get_nnz(); i++) {
    if (get_row_ptr()[i] == get_col_ind()[i]) {
      vec[get_row_ptr()[i]] = get_val_ptr()[i];
    }
  }

  logger.func_out();
}
template void monolish::matrix::COO<double>::diag(vector<double> &vec) const;
template void monolish::matrix::COO<float>::diag(vector<float> &vec) const;

template <typename T> void COO<T>::diag(view1D<vector<T>, T> &vec) const {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  auto s = get_row() > get_col() ? get_col() : get_row();
  assert(s == vec.size());

  for (auto i = decltype(vec.size()){0}; i < vec.size(); i++) {
    vec[i] = 0;
  }

  for (auto i = decltype(vec.size()){0}; i < get_nnz(); i++) {
    if (get_row_ptr()[i] == get_col_ind()[i]) {
      vec[get_row_ptr()[i]] = get_val_ptr()[i];
    }
  }
  logger.func_out();
}
template void
monolish::matrix::COO<double>::diag(view1D<vector<double>, double> &vec) const;
template void
monolish::matrix::COO<float>::diag(view1D<vector<float>, float> &vec) const;

template <typename T>
void COO<T>::diag(view1D<matrix::Dense<T>, T> &vec) const {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  auto s = get_row() > get_col() ? get_col() : get_row();
  assert(s == vec.size());

  for (auto i = decltype(vec.size()){0}; i < vec.size(); i++) {
    vec[i] = 0;
  }

  for (auto i = decltype(vec.size()){0}; i < get_nnz(); i++) {
    if (get_row_ptr()[i] == get_col_ind()[i]) {
      vec[get_row_ptr()[i]] = get_val_ptr()[i];
    }
  }

  logger.func_out();
}
template void monolish::matrix::COO<double>::diag(
    view1D<matrix::Dense<double>, double> &vec) const;
template void monolish::matrix::COO<float>::diag(
    view1D<matrix::Dense<float>, float> &vec) const;

// row
template <typename T> void COO<T>::row(const size_t r, vector<T> &vec) const {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  assert(get_col() == vec.size());

  for (auto i = decltype(vec.size()){0}; i < vec.size(); i++) {
    vec[i] = 0;
  }

  for (auto i = decltype(vec.size()){0}; i < get_nnz(); i++) {
    if (get_row_ptr()[i] == static_cast<int>(r)) {
      vec[get_col_ind()[i]] = get_val_ptr()[i];
    }
  }
  logger.func_out();
}
template void monolish::matrix::COO<double>::row(const size_t r,
                                                 vector<double> &vec) const;
template void monolish::matrix::COO<float>::row(const size_t r,
                                                vector<float> &vec) const;

template <typename T>
void COO<T>::row(const size_t r, view1D<vector<T>, T> &vec) const {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  assert(get_col() == vec.size());

  for (auto i = decltype(vec.size()){0}; i < vec.size(); i++) {
    vec[i] = 0;
  }

  for (auto i = decltype(vec.size()){0}; i < get_nnz(); i++) {
    if (get_row_ptr()[i] == static_cast<int>(r)) {
      vec[get_col_ind()[i]] = get_val_ptr()[i];
    }
  }
  logger.func_out();
}
template void
monolish::matrix::COO<double>::row(const size_t r,
                                   view1D<vector<double>, double> &vec) const;
template void
monolish::matrix::COO<float>::row(const size_t r,
                                  view1D<vector<float>, float> &vec) const;

template <typename T>
void COO<T>::row(const size_t r, view1D<matrix::Dense<T>, T> &vec) const {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  assert(get_col() == vec.size());

  for (auto i = decltype(vec.size()){0}; i < vec.size(); i++) {
    vec[i] = 0;
  }

  for (auto i = decltype(vec.size()){0}; i < get_nnz(); i++) {
    if (get_row_ptr()[i] == static_cast<int>(r)) {
      vec[get_col_ind()[i]] = get_val_ptr()[i];
    }
  }
  logger.func_out();
}
template void monolish::matrix::COO<double>::row(
    const size_t r, view1D<matrix::Dense<double>, double> &vec) const;
template void monolish::matrix::COO<float>::row(
    const size_t r, view1D<matrix::Dense<float>, float> &vec) const;

// col
template <typename T> void COO<T>::col(const size_t c, vector<T> &vec) const {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  assert(get_row() == vec.size());

  for (auto i = decltype(vec.size()){0}; i < vec.size(); i++) {
    vec[i] = 0;
  }

  for (auto i = decltype(vec.size()){0}; i < get_nnz(); i++) {
    if (get_col_ind()[i] == static_cast<int>(c)) {
      vec[get_row_ptr()[i]] = get_val_ptr()[i];
    }
  }
  logger.func_out();
}
template void monolish::matrix::COO<double>::col(const size_t c,
                                                 vector<double> &vec) const;
template void monolish::matrix::COO<float>::col(const size_t c,
                                                vector<float> &vec) const;

template <typename T>
void COO<T>::col(const size_t c, view1D<vector<T>, T> &vec) const {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  assert(get_row() == vec.size());

  for (auto i = decltype(vec.size()){0}; i < vec.size(); i++) {
    vec[i] = 0;
  }

  for (auto i = decltype(vec.size()){0}; i < get_nnz(); i++) {
    if (get_col_ind()[i] == static_cast<int>(c)) {
      vec[get_row_ptr()[i]] = get_val_ptr()[i];
    }
  }
  logger.func_out();
}
template void
monolish::matrix::COO<double>::col(const size_t c,
                                   view1D<vector<double>, double> &vec) const;
template void
monolish::matrix::COO<float>::col(const size_t c,
                                  view1D<vector<float>, float> &vec) const;

template <typename T>
void COO<T>::col(const size_t c, view1D<matrix::Dense<T>, T> &vec) const {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  assert(get_row() == vec.size());

  for (auto i = decltype(vec.size()){0}; i < vec.size(); i++) {
    vec[i] = 0;
  }

  for (auto i = decltype(vec.size()){0}; i < get_nnz(); i++) {
    if (get_col_ind()[i] == static_cast<int>(c)) {
      vec[get_row_ptr()[i]] = get_val_ptr()[i];
    }
  }
  logger.func_out();
}
template void monolish::matrix::COO<double>::col(
    const size_t c, view1D<matrix::Dense<double>, double> &vec) const;
template void monolish::matrix::COO<float>::col(
    const size_t c, view1D<matrix::Dense<float>, float> &vec) const;
} // namespace matrix
} // namespace monolish
