#include "monolish_blas.hpp"
#include "monolish_equation.hpp"
#include <iostream>

// Template a matrix format, solver and preconditioer.
template <typename MATRIX, typename SOLVER, typename PRECOND, typename FLOAT>
void solve() {
  monolish::matrix::COO<FLOAT> A_COO("sample.mtx"); // Input from file

  // Edit the matrix as needed //
  // Execute A_COO.sort() after editing the matrix //

  MATRIX A(A_COO); // Create CRS format and convert from COO format

  // Length A.row()
  // Random vector length A.row() with random values in the range 1.0 to 2.0
  monolish::vector<FLOAT> x(A.get_row(), 1.0, 2.0);
  monolish::vector<FLOAT> b(A.get_row(), 1.0, 2.0);

  // Create solver
  SOLVER solver;

  // Create preconditioner
  PRECOND precond;
  solver.set_create_precond(precond);
  solver.set_apply_precond(precond);

  // Set solver options
  solver.set_tol(1.0e-12);
  solver.set_maxiter(A.get_row());

  // if you need residual history
  // solver.set_print_rhistory(true);
  // solver.set_rhistory_filename("./a.txt");

  // Solve
  if (monolish::util::solver_check(solver.solve(A, x, b))) {
    return;
  }

  // output x to standard output
  x.print_all();
}

int main() {

  // output log if you need
  // monolish::util::set_log_level(3);
  // monolish::util::set_log_filename("./monolish_test_log.txt");

  std::cout
      << "A is CRS, solver is CG, precondition is Jacobi, precision is double"
      << std::endl;
  solve<monolish::matrix::CRS<double>,
        monolish::equation::CG<monolish::matrix::CRS<double>, double>,
        monolish::equation::Jacobi<monolish::matrix::CRS<double>, double>,
        double>();

  std::cout << "A is Dense, solver is BiCGSTAB, precondition is none, "
               "precision is float"
            << std::endl;
  solve<monolish::matrix::Dense<float>,
        monolish::equation::BiCGSTAB<monolish::matrix::Dense<float>, float>,
        monolish::equation::none<monolish::matrix::Dense<float>, float>,
        float>();

  std::cout
      << "A is Dense, solver is LU, precondition is none, precision is double"
      << std::endl;
  solve<monolish::matrix::Dense<double>,
        monolish::equation::LU<monolish::matrix::Dense<double>, double>,
        monolish::equation::none<monolish::matrix::Dense<double>, double>,
        double>();

  return 0;
}
