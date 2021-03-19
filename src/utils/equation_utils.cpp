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
namespace {

  template <typename T, typename V1, typename V2>
    T get_residual_l2_core(const matrix::Dense<T> &A, const V1 &x,
        const V2 &y) {
      Logger &logger = Logger::get_instance();
      logger.util_in(monolish_func);
      vector<T> tmp(x.size());
      tmp.send();

      blas::matvec(A, x, tmp); // tmp=Ax
      vml::sub(y, tmp, tmp);
      logger.util_out();
      return blas::nrm2(tmp);
    }

  template <typename T, typename V1, typename V2>
    T get_residual_l2_core(const matrix::CRS<T> &A, const V1 &x,
        const V2 &y) {
      Logger &logger = Logger::get_instance();
      logger.util_in(monolish_func);
      vector<T> tmp(x.size());
      tmp.send();

      blas::matvec(A, x, tmp); // tmp=Ax
      vml::sub(y, tmp, tmp);
      logger.util_out();
      return blas::nrm2(tmp);
    }

  template <typename T, typename V1, typename V2>
    T get_residual_l2_core(const matrix::LinearOperator<T> &A, const V1 &x,
        const V2 &y) {
      Logger &logger = Logger::get_instance();
      logger.util_in(monolish_func);
      vector<T> tmp(x.size());
      tmp.send();

      blas::matvec(A, x, tmp); // tmp=Ax
      vml::sub(y, tmp, tmp);
      logger.util_out();
      return blas::nrm2(tmp);
    }
}
double get_residual_l2(const matrix::Dense<double> &A, const vector<double> &x,
            const vector<double> &y){return get_residual_l2_core(A, x, y);}
double get_residual_l2(const matrix::Dense<double> &A, const vector<double> &x,
            const view1D<vector<double>, double> &y){return get_residual_l2_core(A, x, y);}
double get_residual_l2(const matrix::Dense<double> &A, const vector<double> &x,
            const view1D<matrix::Dense<double>, double> &y){return get_residual_l2_core(A, x, y);}
double get_residual_l2(const matrix::Dense<double> &A,
            const view1D<vector<double>, double> &x, const vector<double> &y){return get_residual_l2_core(A, x, y);}
double get_residual_l2(const matrix::Dense<double> &A,
            const view1D<vector<double>, double> &x,
            const view1D<vector<double>, double> &y){return get_residual_l2_core(A, x, y);}
double get_residual_l2(const matrix::Dense<double> &A,
            const view1D<vector<double>, double> &x,
            const view1D<matrix::Dense<double>, double> &y){return get_residual_l2_core(A, x, y);}
double get_residual_l2(const matrix::Dense<double> &A,
            const view1D<matrix::Dense<double>, double> &x, const vector<double> &y){return get_residual_l2_core(A, x, y);}
double get_residual_l2(const matrix::Dense<double> &A,
            const view1D<matrix::Dense<double>, double> &x,
            const view1D<vector<double>, double> &y){return get_residual_l2_core(A, x, y);}
double get_residual_l2(const matrix::Dense<double> &A,
            const view1D<matrix::Dense<double>, double> &x,
            const view1D<matrix::Dense<double>, double> &y){return get_residual_l2_core(A, x, y);}
float get_residual_l2(const matrix::Dense<float> &A, const vector<float> &x,
            const vector<float> &y){return get_residual_l2_core(A, x, y);}
float get_residual_l2(const matrix::Dense<float> &A, const vector<float> &x,
            const view1D<vector<float>, float> &y){return get_residual_l2_core(A, x, y);}
float get_residual_l2(const matrix::Dense<float> &A, const vector<float> &x,
            const view1D<matrix::Dense<float>, float> &y){return get_residual_l2_core(A, x, y);}
float get_residual_l2(const matrix::Dense<float> &A,
            const view1D<vector<float>, float> &x, const vector<float> &y){return get_residual_l2_core(A, x, y);}
float get_residual_l2(const matrix::Dense<float> &A,
            const view1D<vector<float>, float> &x,
            const view1D<vector<float>, float> &y){return get_residual_l2_core(A, x, y);}
float get_residual_l2(const matrix::Dense<float> &A,
            const view1D<vector<float>, float> &x,
            const view1D<matrix::Dense<float>, float> &y){return get_residual_l2_core(A, x, y);}
float get_residual_l2(const matrix::Dense<float> &A,
            const view1D<matrix::Dense<float>, float> &x, const vector<float> &y){return get_residual_l2_core(A, x, y);}
float get_residual_l2(const matrix::Dense<float> &A,
            const view1D<matrix::Dense<float>, float> &x,
            const view1D<vector<float>, float> &y){return get_residual_l2_core(A, x, y);}
float get_residual_l2(const matrix::Dense<float> &A,
            const view1D<matrix::Dense<float>, float> &x,
            const view1D<matrix::Dense<float>, float> &y){return get_residual_l2_core(A, x, y);}

double get_residual_l2(const matrix::CRS<double> &A, const vector<double> &x,
            const vector<double> &y){return get_residual_l2_core(A, x, y);}
double get_residual_l2(const matrix::CRS<double> &A, const vector<double> &x,
            const view1D<vector<double>, double> &y){return get_residual_l2_core(A, x, y);}
double get_residual_l2(const matrix::CRS<double> &A, const vector<double> &x,
            const view1D<matrix::Dense<double>, double> &y){return get_residual_l2_core(A, x, y);}
double get_residual_l2(const matrix::CRS<double> &A,
            const view1D<vector<double>, double> &x, const vector<double> &y){return get_residual_l2_core(A, x, y);}
double get_residual_l2(const matrix::CRS<double> &A,
            const view1D<vector<double>, double> &x,
            const view1D<vector<double>, double> &y){return get_residual_l2_core(A, x, y);}
double get_residual_l2(const matrix::CRS<double> &A,
            const view1D<vector<double>, double> &x,
            const view1D<matrix::Dense<double>, double> &y){return get_residual_l2_core(A, x, y);}
double get_residual_l2(const matrix::CRS<double> &A,
            const view1D<matrix::Dense<double>, double> &x, const vector<double> &y){return get_residual_l2_core(A, x, y);}
double get_residual_l2(const matrix::CRS<double> &A,
            const view1D<matrix::Dense<double>, double> &x,
            const view1D<vector<double>, double> &y){return get_residual_l2_core(A, x, y);}
double get_residual_l2(const matrix::CRS<double> &A,
            const view1D<matrix::Dense<double>, double> &x,
            const view1D<matrix::Dense<double>, double> &y){return get_residual_l2_core(A, x, y);}
float get_residual_l2(const matrix::CRS<float> &A, const vector<float> &x,
            const vector<float> &y){return get_residual_l2_core(A, x, y);}
float get_residual_l2(const matrix::CRS<float> &A, const vector<float> &x,
            const view1D<vector<float>, float> &y){return get_residual_l2_core(A, x, y);}
float get_residual_l2(const matrix::CRS<float> &A, const vector<float> &x,
            const view1D<matrix::Dense<float>, float> &y){return get_residual_l2_core(A, x, y);}
float get_residual_l2(const matrix::CRS<float> &A,
            const view1D<vector<float>, float> &x, const vector<float> &y){return get_residual_l2_core(A, x, y);}
float get_residual_l2(const matrix::CRS<float> &A,
            const view1D<vector<float>, float> &x,
            const view1D<vector<float>, float> &y){return get_residual_l2_core(A, x, y);}
float get_residual_l2(const matrix::CRS<float> &A,
            const view1D<vector<float>, float> &x,
            const view1D<matrix::Dense<float>, float> &y){return get_residual_l2_core(A, x, y);}
float get_residual_l2(const matrix::CRS<float> &A,
            const view1D<matrix::Dense<float>, float> &x, const vector<float> &y){return get_residual_l2_core(A, x, y);}
float get_residual_l2(const matrix::CRS<float> &A,
            const view1D<matrix::Dense<float>, float> &x,
            const view1D<vector<float>, float> &y){return get_residual_l2_core(A, x, y);}
float get_residual_l2(const matrix::CRS<float> &A,
            const view1D<matrix::Dense<float>, float> &x,
            const view1D<matrix::Dense<float>, float> &y){return get_residual_l2_core(A, x, y);}

double get_residual_l2(const matrix::LinearOperator<double> &A, const vector<double> &x,
            const vector<double> &y){return get_residual_l2_core(A, x, y);}
double get_residual_l2(const matrix::LinearOperator<double> &A, const vector<double> &x,
            const view1D<vector<double>, double> &y){return get_residual_l2_core(A, x, y);}
double get_residual_l2(const matrix::LinearOperator<double> &A, const vector<double> &x,
            const view1D<matrix::Dense<double>, double> &y){return get_residual_l2_core(A, x, y);}
double get_residual_l2(const matrix::LinearOperator<double> &A,
            const view1D<vector<double>, double> &x, const vector<double> &y){return get_residual_l2_core(A, x, y);}
double get_residual_l2(const matrix::LinearOperator<double> &A,
            const view1D<vector<double>, double> &x,
            const view1D<vector<double>, double> &y){return get_residual_l2_core(A, x, y);}
double get_residual_l2(const matrix::LinearOperator<double> &A,
            const view1D<vector<double>, double> &x,
            const view1D<matrix::Dense<double>, double> &y){return get_residual_l2_core(A, x, y);}
double get_residual_l2(const matrix::LinearOperator<double> &A,
            const view1D<matrix::Dense<double>, double> &x, const vector<double> &y){return get_residual_l2_core(A, x, y);}
double get_residual_l2(const matrix::LinearOperator<double> &A,
            const view1D<matrix::Dense<double>, double> &x,
            const view1D<vector<double>, double> &y){return get_residual_l2_core(A, x, y);}
double get_residual_l2(const matrix::LinearOperator<double> &A,
            const view1D<matrix::Dense<double>, double> &x,
            const view1D<matrix::Dense<double>, double> &y){return get_residual_l2_core(A, x, y);}
float get_residual_l2(const matrix::LinearOperator<float> &A, const vector<float> &x,
            const vector<float> &y){return get_residual_l2_core(A, x, y);}
float get_residual_l2(const matrix::LinearOperator<float> &A, const vector<float> &x,
            const view1D<vector<float>, float> &y){return get_residual_l2_core(A, x, y);}
float get_residual_l2(const matrix::LinearOperator<float> &A, const vector<float> &x,
            const view1D<matrix::Dense<float>, float> &y){return get_residual_l2_core(A, x, y);}
float get_residual_l2(const matrix::LinearOperator<float> &A,
            const view1D<vector<float>, float> &x, const vector<float> &y){return get_residual_l2_core(A, x, y);}
float get_residual_l2(const matrix::LinearOperator<float> &A,
            const view1D<vector<float>, float> &x,
            const view1D<vector<float>, float> &y){return get_residual_l2_core(A, x, y);}
float get_residual_l2(const matrix::LinearOperator<float> &A,
            const view1D<vector<float>, float> &x,
            const view1D<matrix::Dense<float>, float> &y){return get_residual_l2_core(A, x, y);}
float get_residual_l2(const matrix::LinearOperator<float> &A,
            const view1D<matrix::Dense<float>, float> &x, const vector<float> &y){return get_residual_l2_core(A, x, y);}
float get_residual_l2(const matrix::LinearOperator<float> &A,
            const view1D<matrix::Dense<float>, float> &x,
            const view1D<vector<float>, float> &y){return get_residual_l2_core(A, x, y);}
float get_residual_l2(const matrix::LinearOperator<float> &A,
            const view1D<matrix::Dense<float>, float> &x,
            const view1D<matrix::Dense<float>, float> &y){return get_residual_l2_core(A, x, y);}

} // namespace monolish
