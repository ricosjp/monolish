#include "../../include/monolish_blas.hpp"
#include "../../include/monolish_vml.hpp"
#include "../internal/monolish_internal.hpp"

namespace monolish {

bool util::solver_check(const int err) {
  switch (err) {
  case MONOLISH_SOLVER_SUCCESS:
    return 0;
  case MONOLISH_SOLVER_MAXITER:
    std::runtime_error("equation error, maxiter\n");
    return false;
  case MONOLISH_SOLVER_BREAKDOWN:
    std::runtime_error("equation error, breakdown\n");
    return false;
  case MONOLISH_SOLVER_SIZE_ERROR:
    std::runtime_error("equation error, size error\n");
    return false;
  case MONOLISH_SOLVER_RESIDUAL_NAN:
    std::runtime_error("equation error, resudual is nan\n");
    return false;
  case MONOLISH_SOLVER_NOT_IMPL:
    std::runtime_error("equation error, this solver is not impl.\n");
    return false;
  default:
    return 0;
  }
}

template <typename T>
T util::get_residual_l2(const matrix::CRS<T> &A, const vector<T> &x, const vector<T> &b) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  vector<T> tmp(x.size());
  tmp.send();

  blas::matvec(A, x, tmp); // tmp=Ax
  vml::sub(b, tmp, tmp);
  logger.util_out();
  return blas::nrm2(tmp);
}
template double util::get_residual_l2(const matrix::CRS<double> &A, const vector<double> &x,
                                      const vector<double> &b);
template float util::get_residual_l2(const matrix::CRS<float> &A, const vector<float> &x,
                                     const vector<float> &b);
template <typename T>
T util::get_residual_l2(const matrix::Dense<T> &A, const vector<T> &x, const vector<T> &b) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  vector<T> tmp(x.size());
  tmp.send();

  blas::matvec(A, x, tmp); // tmp=Ax
  vml::sub(b, tmp, tmp);
  logger.util_out();
  return blas::nrm2(tmp);
}
template double util::get_residual_l2(const matrix::Dense<double> &A,
                                      const vector<double> &x, const vector<double> &b);
template float util::get_residual_l2(const matrix::Dense<float> &A, const vector<float> &x,
                                     const vector<float> &b);

template <typename T>
T util::get_residual_l2(const matrix::LinearOperator<T> &A, const vector<T> &x,
                        const vector<T> &b) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  vector<T> tmp(x.size());
  tmp.send();

  blas::matvec(A, x, tmp); // tmp=Ax
  vml::sub(b, tmp, tmp);
  logger.util_out();
  return blas::nrm2(tmp);
}
template double util::get_residual_l2(const matrix::LinearOperator<double> &A,
                                      const vector<double> &x, const vector<double> &b);
template float util::get_residual_l2(const matrix::LinearOperator<float> &A,
                                     const vector<float> &x, const vector<float> &b);

} // namespace monolish
