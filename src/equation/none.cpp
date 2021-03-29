#include "../../include/monolish_blas.hpp"
#include "../../include/monolish_equation.hpp"
#include "../internal/monolish_internal.hpp"

namespace monolish {

////////////////////////////////////////
// precond none /////////////////////////
////////////////////////////////////////
template <typename MATRIX, typename T>
void equation::none<MATRIX, T>::create_precond(MATRIX &A) {
  Logger &logger = Logger::get_instance();

  // nothing to do
  (void)(&A);

  logger.solver_in(monolish_func);
  logger.solver_out();
}
template void equation::none<matrix::Dense<float>, float>::create_precond(
    matrix::Dense<float> &A);
template void equation::none<matrix::Dense<double>, double>::create_precond(
    matrix::Dense<double> &A);

template void equation::none<matrix::CRS<float>, float>::create_precond(
    matrix::CRS<float> &A);
template void equation::none<matrix::CRS<double>, double>::create_precond(
    matrix::CRS<double> &A);

template <>
void equation::none<matrix::LinearOperator<float>, float>::create_precond(
    matrix::LinearOperator<float> &A){};
template <>
void equation::none<matrix::LinearOperator<double>, double>::create_precond(
    matrix::LinearOperator<double> &A){};

/////

template <typename MATRIX, typename T>
void equation::none<MATRIX, T>::apply_precond(const vector<T> &r,
                                              vector<T> &z) {
  Logger &logger = Logger::get_instance();
  logger.solver_in(monolish_func);
  blas::copy(r, z);
  logger.solver_out();
}
template void
equation::none<matrix::Dense<float>, float>::apply_precond(const vector<float> &r,
                                                         vector<float> &z);
template void equation::none<matrix::Dense<double>, double>::apply_precond(
    const vector<double> &r, vector<double> &z);

template void
equation::none<matrix::CRS<float>, float>::apply_precond(const vector<float> &r,
                                                         vector<float> &z);
template void equation::none<matrix::CRS<double>, double>::apply_precond(
    const vector<double> &r, vector<double> &z);

template void
equation::none<matrix::LinearOperator<float>, float>::apply_precond(const vector<float> &r,
                                                         vector<float> &z);
template void equation::none<matrix::LinearOperator<double>, double>::apply_precond(
    const vector<double> &r, vector<double> &z);

/////

template <typename MATRIX, typename T>
int equation::none<MATRIX, T>::solve(MATRIX &A, vector<T> &x, vector<T> &b) {
  Logger &logger = Logger::get_instance();
  logger.solver_in(monolish_func);
  // nothing to do
  (void)(&A);
  (void)(&x);
  (void)(&b);
  logger.solver_out();
  return MONOLISH_SOLVER_SUCCESS;
}
template int equation::none<matrix::Dense<float>, float>::solve(
    matrix::Dense<float> &A, vector<float> &x, vector<float> &b);
template int equation::none<matrix::Dense<double>, double>::solve(
    matrix::Dense<double> &A, vector<double> &x, vector<double> &b);

template int equation::none<matrix::CRS<float>, float>::solve(
    matrix::CRS<float> &A, vector<float> &x, vector<float> &b);
template int equation::none<matrix::CRS<double>, double>::solve(
    matrix::CRS<double> &A, vector<double> &x, vector<double> &b);

template int equation::none<matrix::LinearOperator<float>, float>::solve(
    matrix::LinearOperator<float> &A, vector<float> &x, vector<float> &b);
template int equation::none<matrix::LinearOperator<double>, double>::solve(
    matrix::LinearOperator<double> &A, vector<double> &x, vector<double> &b);
} // namespace monolish
