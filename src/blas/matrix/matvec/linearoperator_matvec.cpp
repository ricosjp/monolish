#include "../../../../include/monolish_blas.hpp"
#include "../../../internal/monolish_internal.hpp"

namespace monolish {

void blas::matvec(const matrix::LinearOperator<double> &A,
                  const vector<double> &x, vector<double> &y) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err, M = MN * N
  if (A.get_row() != y.size() && A.get_col() != x.size()) {
    throw std::runtime_error("error vector size is not same");
  }

  if (A.get_device_mem_stat() != x.get_device_mem_stat() ||
      A.get_device_mem_stat() != y.get_device_mem_stat()) {
    throw std::runtime_error("error get_device_mem_stat() is not same");
  }

  if (!A.get_matvec_init_flag()) {
    throw std::runtime_error("matvec is not defined in A");
  }

  y = A.get_matvec()(x);

  logger.func_out();
}

void blas::rmatvec(const matrix::LinearOperator<double> &A,
                   const vector<double> &x, vector<double> &y) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err, M = MN * N
  if (A.get_row() != y.size() && A.get_col() != x.size()) {
    throw std::runtime_error("error vector size is not same");
  }

  if (A.get_device_mem_stat() != x.get_device_mem_stat() ||
      A.get_device_mem_stat() != y.get_device_mem_stat()) {
    throw std::runtime_error("error get_device_mem_stat() is not same");
  }

  if (!A.get_rmatvec_init_flag()) {
    throw std::runtime_error("rmatvec is not defined in A");
  }

  y = A.get_rmatvec()(x);

  logger.func_out();
}

void blas::matvec(const matrix::LinearOperator<float> &A,
                  const vector<float> &x, vector<float> &y) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err, M = MN * N
  if (A.get_row() != y.size() && A.get_col() != x.size()) {
    throw std::runtime_error("error vector size is not same");
  }

  if (A.get_device_mem_stat() != x.get_device_mem_stat() ||
      A.get_device_mem_stat() != y.get_device_mem_stat()) {
    throw std::runtime_error("error get_device_mem_stat() is not same");
  }

  if (!A.get_matvec_init_flag()) {
    throw std::runtime_error("matvec is not defined in A");
  }

  y = A.get_matvec()(x);

  logger.func_out();
}

void blas::rmatvec(const matrix::LinearOperator<float> &A,
                   const vector<float> &x, vector<float> &y) {
  Logger &logger = Logger::get_instance();
  logger.func_in(monolish_func);

  // err, M = MN * N
  if (A.get_row() != y.size() && A.get_col() != x.size()) {
    throw std::runtime_error("error vector size is not same");
  }

  if (A.get_device_mem_stat() != x.get_device_mem_stat() ||
      A.get_device_mem_stat() != y.get_device_mem_stat()) {
    throw std::runtime_error("error get_device_mem_stat() is not same");
  }

  if (!A.get_rmatvec_init_flag()) {
    throw std::runtime_error("rmatvec is not defined in A");
  }

  y = A.get_rmatvec()(x);

  logger.func_out();
}

} // namespace monolish
