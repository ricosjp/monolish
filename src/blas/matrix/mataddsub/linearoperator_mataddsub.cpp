#include "../../../../include/monolish_blas.hpp"
#include "../../../internal/monolish_internal.hpp"

namespace monolish {

namespace {

template <typename F>
void matadd_core(const matrix::LinearOperator<F> &A,
                 const matrix::LinearOperator<F> &B,
                 matrix::LinearOperator<F> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(A, B, C));
  assert(util::is_same_device_mem_stat(A, B, C));

  assert(A.get_matvec_init_flag() == B.get_matvec_init_flag());
  assert(A.get_rmatvec_init_flag() == B.get_rmatvec_init_flag());

  if (A.get_matvec_init_flag()) {
    C.set_matvec([&](const vector<F> &VEC) {
      vector<F> vec(A.get_row(), 0.0), vec_tmp(A.get_row(), 0.0);
      if (A.get_device_mem_stat()) {
        util::send(vec, vec_tmp);
      }
      blas::matvec(A, VEC, vec);
      blas::matvec(B, VEC, vec_tmp);
      blas::axpy(1.0, vec_tmp, vec);
      if (A.get_device_mem_stat()) {
        util::device_free(vec_tmp);
      }
      return vec;
    });
  }

  if (A.get_rmatvec_init_flag()) {
    C.set_rmatvec([&](const vector<F> &VEC) {
      vector<F> vec(A.get_col(), 0.0), vec_tmp(A.get_col(), 0.0);
      if (A.get_device_mem_stat()) {
        util::send(vec, vec_tmp);
      }
      blas::rmatvec(A, VEC, vec);
      blas::rmatvec(B, VEC, vec_tmp);
      blas::axpy(1.0, vec_tmp, vec);
      if (A.get_device_mem_stat()) {
        util::device_free(vec_tmp);
      }
      return vec;
    });
  }

  logger.func_out();
}

template <typename F>
void matsub_core(const matrix::LinearOperator<F> &A,
                 const matrix::LinearOperator<F> &B,
                 matrix::LinearOperator<F> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(A, B, C));
  assert(util::is_same_device_mem_stat(A, B, C));

  assert(A.get_matvec_init_flag() == B.get_matvec_init_flag());
  assert(A.get_rmatvec_init_flag() == B.get_rmatvec_init_flag());

  if (A.get_matvec_init_flag()) {
    C.set_matvec([&](const vector<F> &VEC) {
      vector<F> vec(A.get_row(), 0.0), vec_tmp(A.get_row(), 0.0);
      if (A.get_device_mem_stat()) {
        util::send(vec, vec_tmp);
      }
      blas::matvec(A, VEC, vec);
      blas::matvec(B, VEC, vec_tmp);
      blas::axpy(-1.0, vec_tmp, vec);
      if (A.get_device_mem_stat()) {
        util::device_free(vec_tmp);
      }
      return vec;
    });
  }

  if (A.get_rmatvec_init_flag()) {
    C.set_rmatvec([&](const vector<F> &VEC) {
      vector<F> vec(A.get_col(), 0.0), vec_tmp(A.get_col(), 0.0);
      if (A.get_device_mem_stat()) {
        util::send(vec, vec_tmp);
      }
      blas::rmatvec(A, VEC, vec);
      blas::rmatvec(B, VEC, vec_tmp);
      blas::axpy(-1.0, vec_tmp, vec);
      if (A.get_device_mem_stat()) {
        util::device_free(vec_tmp);
      }
      return vec;
    });
  }

  logger.func_out();
}
} // namespace

namespace blas {
void matadd(const matrix::LinearOperator<double> &A,
            const matrix::LinearOperator<double> &B,
            matrix::LinearOperator<double> &C) {
  matadd_core(A, B, C);
}
void matsub(const matrix::LinearOperator<double> &A,
            const matrix::LinearOperator<double> &B,
            matrix::LinearOperator<double> &C) {
  matsub_core(A, B, C);
}
void matadd(const matrix::LinearOperator<float> &A,
            const matrix::LinearOperator<float> &B,
            matrix::LinearOperator<float> &C) {
  matadd_core(A, B, C);
}
void matsub(const matrix::LinearOperator<float> &A,
            const matrix::LinearOperator<float> &B,
            matrix::LinearOperator<float> &C) {
  matsub_core(A, B, C);
}

} // namespace blas
} // namespace monolish
