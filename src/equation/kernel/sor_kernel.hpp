#pragma once

namespace monolish {
namespace {
template <typename T>
void sor_kernel_lower(const monolish::matrix::CRS<T> &A, const vector<T> &D,
                      vector<T> &x, const vector<T> &b) {

  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);
  for (int i = 0; i < (int)A.get_row(); i++) {
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
  logger.func_out();
}

template <typename T>
void sor_kernel_lower(const monolish::matrix::Dense<T> &A, const vector<T> &D,
                      vector<T> &x, const vector<T> &b) {

  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);
  for (int i = 0; i < (int)A.get_row(); i++) {
    auto tmp = b.data()[i];
    for (int j = 0; j < i; j++) {
      tmp -= A.val[i * A.get_col() + j] * x.data()[j];
    }
    x.data()[i] = tmp * D.data()[i];
  }
  logger.func_out();
}

template <typename T>
void sor_kernel_precond(const monolish::matrix::CRS<T> &A, const vector<T> &D,
        vector<T> &x, const vector<T> &b) {

    Logger &logger = Logger::get_instance();
    logger.func_in(monolish_func);

    vector<T> b_tmp(b);
    b_tmp.nonfree_recv();

    for (int i = 0; i < (int)A.get_row(); i++) {
        auto tmp = b_tmp.data()[i];
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; j++) {
            if (i > A.col_ind[j]) { // lower
                tmp -= A.val[j] * x.data()[A.col_ind[j]];
            }
        }
            x.data()[i]   = tmp * D.data()[i];
    }
    for (int i = (int)A.get_row() -1; i >= 0; i--) {
        auto tmp = 0.0;
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; j++) {
            if (i < A.col_ind[j] ){ //upper
                tmp += A.val[j] * x.data()[A.col_ind[j]];
                
            }
        }
            x.data()[i]  -= tmp * D.data()[i];
    }
    x.send();

    logger.func_out();
}

template <typename T>
void sor_kernel_precond(const monolish::matrix::Dense<T> &A, const vector<T> &D,
        vector<T> &x, const vector<T> &b) {

    Logger &logger = Logger::get_instance();
    logger.func_in(monolish_func);

    vector<T> b_tmp(b);
    b_tmp.nonfree_recv();

    for (int i = 0; i < (int)A.get_row(); i++) {
        auto tmp = b_tmp.data()[i];
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; j++) {
            if (i > A.col_ind[j]) { // lower
                tmp -= A.val[i * A.get_col() + j] * x.data()[j];
            }
        }
            x.data()[i]   = tmp * D.data()[i];
    }
    for (int i = (int)A.get_row() -1; i >= 0; i--) {
        auto tmp = 0.0;
        for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; j++) {
            if (i < A.col_ind[j] ){ //upper
                tmp += A.val[i * A.get_col() + j] * x.data()[j];
                
            }
        }
            x.data()[i]  -= tmp * D.data()[i];
    }
    x.send();

    logger.func_out();
}
} // namespace
} // namespace monolish
