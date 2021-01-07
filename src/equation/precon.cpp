#include "../../include/monolish_blas.hpp"
#include "../../include/monolish_equation.hpp"
#include "../internal/monolish_internal.hpp"

namespace monolish {

////////////////////////////////////////
// precond none /////////////////////////
////////////////////////////////////////
template <typename T>
void equation::none<T>::create_precond(matrix::CRS<T> &A) {
  Logger &logger = Logger::get_instance();

  // nothing to do
  (void)(&A);

  logger.solver_in(monolish_func);
  logger.solver_out();
}
template void equation::none<float>::create_precond(matrix::CRS<float> &A);
template void equation::none<double>::create_precond(matrix::CRS<double> &A);

/////

template <typename T>
void equation::none<T>::apply_precond(const vector<T> &r, vector<T> &z) {
  Logger &logger = Logger::get_instance();
  logger.solver_in(monolish_func);
  z = r;
  logger.solver_out();
}
template void equation::none<float>::apply_precond(const vector<float> &r,
                                                   vector<float> &z);
template void equation::none<double>::apply_precond(const vector<double> &r,
                                                    vector<double> &z);

/////

template <typename T>
int equation::none<T>::solve(matrix::CRS<T> &A, vector<T> &x, vector<T> &b) {
  Logger &logger = Logger::get_instance();
  logger.solver_in(monolish_func);
  // nothing to do
  (void)(&A);
  (void)(&x);
  (void)(&b);
  logger.solver_out();
  return MONOLISH_SOLVER_SUCCESS;
}
template int equation::none<float>::solve(matrix::CRS<float> &A,
                                          vector<float> &x, vector<float> &b);
template int equation::none<double>::solve(matrix::CRS<double> &A,
                                           vector<double> &x,
                                           vector<double> &b);

//////////////////////////////////////////////////////
// solver set precond /////////////////////////////////
//////////////////////////////////////////////////////
template <typename T>
template <class PRECOND>
void solver::solver<T>::set_create_precond(PRECOND &p) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  precond.create_precond =
      std::bind(&PRECOND::create_precond, &p, std::placeholders::_1);
  logger.util_out();
}

template void
solver::solver<double>::set_create_precond(equation::none<double> &p);
template void
solver::solver<float>::set_create_precond(equation::none<float> &p);
template void
solver::solver<double>::set_create_precond(equation::Jacobi<double> &p);
template void
solver::solver<float>::set_create_precond(equation::Jacobi<float> &p);

/////
template <typename T>
template <class PRECOND>
void solver::solver<T>::set_apply_precond(PRECOND &p) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  precond.apply_precond =
      std::bind(&PRECOND::apply_precond, &p, std::placeholders::_1,
                std::placeholders::_2);
  logger.util_out();
}

template void
solver::solver<double>::set_apply_precond(equation::none<double> &p);
template void
solver::solver<float>::set_apply_precond(equation::none<float> &p);
template void
solver::solver<double>::set_apply_precond(equation::Jacobi<double> &p);
template void
solver::solver<float>::set_apply_precond(equation::Jacobi<float> &p);
} // namespace monolish
