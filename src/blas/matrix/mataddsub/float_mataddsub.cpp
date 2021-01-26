#include "../../../../include/monolish_blas.hpp"
#include "../../../internal/monolish_internal.hpp"

namespace monolish {

///////////////////////////////
// addition
///////////////////////////////

// Dense ///////////////////
void blas::matadd(const matrix::Dense<float> &A,
                  const matrix::Dense<float> &B, matrix::Dense<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(A, B, C));
  assert(util::is_same_device_mem_stat(A, B, C));

  internal::vadd(A.get_nnz(), A.val.data(), B.val.data(), C.val.data(),
                 A.get_device_mem_stat());

  logger.func_out();
}

// CRS ///////////////////
void blas::matadd(const matrix::CRS<float> &A, const matrix::CRS<float> &B,
                  matrix::CRS<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(A, B, C));
  assert(util::is_same_structure(A, B, C));
  assert(util::is_same_device_mem_stat(A, B, C));

  internal::vadd(A.get_nnz(), A.val.data(), B.val.data(), C.val.data(),
                 A.get_device_mem_stat());

  logger.func_out();
}

// LinearOperator ////////////////////////
void blas::matadd(const matrix::LinearOperator<float> &A,
                  const matrix::LinearOperator<float> &B,
                  matrix::LinearOperator<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(A, B, C));
  assert(util::is_same_device_mem_stat(A, B, C));

  assert(A.get_matvec_init_flag() == B.get_matvec_init_flag());
  assert(A.get_rmatvec_init_flag() == B.get_rmatvec_init_flag());

  if (A.get_matvec_init_flag()) {
    C.set_matvec([&](const vector<float> &VEC) {
      vector<float> vec(A.get_row(), 0.0), vec_tmp(A.get_row(), 0.0);
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
    C.set_rmatvec([&](const vector<float> &VEC) {
      vector<float> vec(A.get_col(), 0.0), vec_tmp(A.get_col(), 0.0);
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

///////////////////////////////
// subtract
///////////////////////////////

// Dense ///////////////////
void blas::matsub(const matrix::Dense<float> &A,
                  const matrix::Dense<float> &B, matrix::Dense<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(A, B, C));
  assert(util::is_same_device_mem_stat(A, B, C));

  internal::vsub(A.get_nnz(), A.val.data(), B.val.data(), C.val.data(),
                 A.get_device_mem_stat());

  logger.func_out();
}

// CRS ///////////////////
void blas::matsub(const matrix::CRS<float> &A, const matrix::CRS<float> &B,
                  matrix::CRS<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(A, B, C));
  assert(util::is_same_structure(A, B, C));
  assert(util::is_same_device_mem_stat(A, B, C));

  internal::vsub(A.get_nnz(), A.val.data(), B.val.data(), C.val.data(),
                 A.get_device_mem_stat());

  logger.func_out();
}

// LinearOperator ///////
void blas::matsub(const matrix::LinearOperator<float> &A,
                  const matrix::LinearOperator<float> &B,
                  matrix::LinearOperator<float> &C) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err
  assert(util::is_same_size(A, B, C));
  assert(util::is_same_device_mem_stat(A, B, C));

  assert(A.get_matvec_init_flag() == B.get_matvec_init_flag());
  assert(A.get_rmatvec_init_flag() == B.get_rmatvec_init_flag());

  if (A.get_matvec_init_flag()) {
    C.set_matvec([&](const vector<float> &VEC) {
      vector<float> vec(A.get_row(), 0.0), vec_tmp(A.get_row(), 0.0);
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
    C.set_rmatvec([&](const vector<float> &VEC) {
      vector<float> vec(A.get_col(), 0.0), vec_tmp(A.get_col(), 0.0);
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

} // namespace monolish
