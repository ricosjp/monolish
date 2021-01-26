#include "../../../../include/monolish_blas.hpp"
#include "../../../internal/monolish_internal.hpp"

namespace monolish {

void blas::matvec(const matrix::LinearOperator<double> &A,
                  const vector<double> &x, vector<double> &y) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err, M = MN * N
  assert(A.get_row() == y.size());
  assert(A.get_col() == x.size());
  assert(util::is_same_device_mem_stat(A, x, y));
  assert(A.get_matvec_init_flag());

  y = A.get_matvec()(x);

  logger.func_out();
}

void blas::rmatvec(const matrix::LinearOperator<double> &A,
                   const vector<double> &x, vector<double> &y) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err, M = MN * N
  assert(A.get_row() == y.size());
  assert(A.get_col() == x.size());
  assert(util::is_same_device_mem_stat(A, x, y));
  assert(A.get_rmatvec_init_flag());

  y = A.get_rmatvec()(x);

  logger.func_out();
}

void blas::matvec(const matrix::LinearOperator<float> &A,
                  const vector<float> &x, vector<float> &y) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err, M = MN * N
  assert(A.get_row() == y.size());
  assert(A.get_col() == x.size());
  assert(util::is_same_device_mem_stat(A, x, y));
  assert(A.get_matvec_init_flag());

  y = A.get_matvec()(x);

  logger.func_out();
}

void blas::rmatvec(const matrix::LinearOperator<float> &A,
                   const vector<float> &x, vector<float> &y) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err, M = MN * N
  assert(A.get_row() == y.size());
  assert(A.get_col() == x.size());
  assert(util::is_same_device_mem_stat(A, x, y));
  assert(A.get_rmatvec_init_flag());

  y = A.get_rmatvec()(x);

  logger.func_out();
}

} // namespace monolish
