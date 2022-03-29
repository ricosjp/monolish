#include "../../../../include/monolish_blas.hpp"
#include "../../../internal/monolish_internal.hpp"

namespace monolish {

// double ////////////////
void blas::matmul(const matrix::LinearOperator<double> &A,
                  const matrix::LinearOperator<double> &B,
                  matrix::LinearOperator<double> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(A.get_col() == B.get_row());
  assert(A.get_row() == C.get_row());
  assert(B.get_col() == C.get_col());
  assert(util::is_same_device_mem_stat(A, B, C));

  assert(A.get_matvec_init_flag() == B.get_matvec_init_flag());
  assert(A.get_rmatvec_init_flag() == B.get_rmatvec_init_flag());

  if (A.get_matvec_init_flag()) {
    C.set_matvec([&](const vector<double> &VEC) {
      vector<double> vec(A.get_row(), 0.0), vec_tmp(B.get_row(), 0.0);
      if (A.get_device_mem_stat()) {
        util::send(vec, vec_tmp);
      }
      blas::matvec(B, VEC, vec_tmp);
      blas::matvec(A, vec_tmp, vec);
      if (A.get_device_mem_stat()) {
        util::device_free(vec_tmp);
      }
      return vec;
    });
  }
  if (A.get_rmatvec_init_flag()) {
    C.set_rmatvec([&](const vector<double> &VEC) {
      vector<double> vec(B.get_col(), 0.0), vec_tmp(A.get_col(), 0.0);
      if (A.get_device_mem_stat()) {
        util::send(vec, vec_tmp);
      }
      blas::rmatvec(A, VEC, vec_tmp);
      blas::rmatvec(B, vec_tmp, vec);
      if (A.get_device_mem_stat()) {
        util::device_free(vec_tmp);
      }
      return vec;
    });
  }

  logger.func_out();
}

// float ////////////////
void blas::matmul(const matrix::LinearOperator<float> &A,
                  const matrix::LinearOperator<float> &B,
                  matrix::LinearOperator<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(A.get_col() == B.get_row());
  assert(A.get_row() == C.get_row());
  assert(B.get_col() == C.get_col());
  assert(util::is_same_device_mem_stat(A, B, C));

  assert(A.get_matvec_init_flag() == B.get_matvec_init_flag());
  assert(A.get_rmatvec_init_flag() == B.get_rmatvec_init_flag());

  if (A.get_matvec_init_flag()) {
    C.set_matvec([&](const vector<float> &VEC) {
      vector<float> vec(A.get_row(), 0.0), vec_tmp(B.get_row(), 0.0);
      if (A.get_device_mem_stat()) {
        util::send(vec, vec_tmp);
      }
      blas::matvec(B, VEC, vec_tmp);
      blas::matvec(A, vec_tmp, vec);
      if (A.get_device_mem_stat()) {
        util::device_free(vec_tmp);
      }
      return vec;
    });
  }
  if (A.get_rmatvec_init_flag()) {
    C.set_rmatvec([&](const vector<float> &VEC) {
      vector<float> vec(B.get_col(), 0.0), vec_tmp(A.get_col(), 0.0);
      if (A.get_device_mem_stat()) {
        util::send(vec, vec_tmp);
      }
      blas::rmatvec(A, VEC, vec_tmp);
      blas::rmatvec(B, vec_tmp, vec);
      if (A.get_device_mem_stat()) {
        util::device_free(vec_tmp);
      }
      return vec;
    });
  }

  logger.func_out();
}
} // namespace monolish
