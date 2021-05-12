#include "../../include/monolish_blas.hpp"
#include "../../include/monolish_equation.hpp"
#include "../internal/monolish_internal.hpp"

namespace monolish {
//////////////////////////////////////////////////////
// solver set precond /////////////////////////////////
//////////////////////////////////////////////////////
template <typename MATRIX, typename T>
template <class PRECOND>
void solver::solver<MATRIX, T>::set_create_precond(PRECOND &p) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  precond.create_precond =
      std::bind(&PRECOND::create_precond, &p, std::placeholders::_1);
  logger.util_out();
}

// none
template void solver::solver<matrix::Dense<double>, double>::set_create_precond(
    equation::none<matrix::Dense<double>, double> &p);
template void solver::solver<matrix::Dense<float>, float>::set_create_precond(
    equation::none<matrix::Dense<float>, float> &p);

template void solver::solver<matrix::CRS<double>, double>::set_create_precond(
    equation::none<matrix::CRS<double>, double> &p);
template void solver::solver<matrix::CRS<float>, float>::set_create_precond(
    equation::none<matrix::CRS<float>, float> &p);

template void
solver::solver<matrix::LinearOperator<double>, double>::set_create_precond(
    equation::none<matrix::LinearOperator<double>, double> &p);
template void
solver::solver<matrix::LinearOperator<float>, float>::set_create_precond(
    equation::none<matrix::LinearOperator<float>, float> &p);

// jacobi
template void solver::solver<matrix::Dense<double>, double>::set_create_precond(
    equation::Jacobi<matrix::Dense<double>, double> &p);
template void solver::solver<matrix::Dense<float>, float>::set_create_precond(
    equation::Jacobi<matrix::Dense<float>, float> &p);

template void solver::solver<matrix::CRS<double>, double>::set_create_precond(
    equation::Jacobi<matrix::CRS<double>, double> &p);
template void solver::solver<matrix::CRS<float>, float>::set_create_precond(
    equation::Jacobi<matrix::CRS<float>, float> &p);

template void
solver::solver<matrix::LinearOperator<double>, double>::set_create_precond(
    equation::Jacobi<matrix::LinearOperator<double>, double> &p);
template void
solver::solver<matrix::LinearOperator<float>, float>::set_create_precond(
    equation::Jacobi<matrix::LinearOperator<float>, float> &p);

/////
template <typename MATRIX, typename T>
template <class PRECOND>
void solver::solver<MATRIX, T>::set_apply_precond(PRECOND &p) {
  Logger &logger = Logger::get_instance();
  logger.util_in(monolish_func);
  precond.apply_precond =
      std::bind(&PRECOND::apply_precond, &p, std::placeholders::_1,
                std::placeholders::_2);
  logger.util_out();
}
// none
template void solver::solver<matrix::Dense<double>, double>::set_apply_precond(
    equation::none<matrix::Dense<double>, double> &p);
template void solver::solver<matrix::Dense<float>, float>::set_apply_precond(
    equation::none<matrix::Dense<float>, float> &p);

template void solver::solver<matrix::CRS<double>, double>::set_apply_precond(
    equation::none<matrix::CRS<double>, double> &p);
template void solver::solver<matrix::CRS<float>, float>::set_apply_precond(
    equation::none<matrix::CRS<float>, float> &p);

template void
solver::solver<matrix::LinearOperator<double>, double>::set_apply_precond(
    equation::none<matrix::LinearOperator<double>, double> &p);
template void
solver::solver<matrix::LinearOperator<float>, float>::set_apply_precond(
    equation::none<matrix::LinearOperator<float>, float> &p);

// jacobi
template void solver::solver<matrix::Dense<double>, double>::set_apply_precond(
    equation::Jacobi<matrix::Dense<double>, double> &p);
template void solver::solver<matrix::Dense<float>, float>::set_apply_precond(
    equation::Jacobi<matrix::Dense<float>, float> &p);

template void solver::solver<matrix::CRS<double>, double>::set_apply_precond(
    equation::Jacobi<matrix::CRS<double>, double> &p);
template void solver::solver<matrix::CRS<float>, float>::set_apply_precond(
    equation::Jacobi<matrix::CRS<float>, float> &p);

template void
solver::solver<matrix::LinearOperator<double>, double>::set_apply_precond(
    equation::Jacobi<matrix::LinearOperator<double>, double> &p);
template void
solver::solver<matrix::LinearOperator<float>, float>::set_apply_precond(
    equation::Jacobi<matrix::LinearOperator<float>, float> &p);
} // namespace monolish
