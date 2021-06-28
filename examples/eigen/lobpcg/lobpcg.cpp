#include <iostream>
#include <monolish_eigen.hpp>
#include <monolish_equation.hpp>

int main() {

  // Output log if you need
  // monolish::util::set_log_level(3);
  // monolish::util::set_log_filename("./monolish_test_log.txt");

  // Create tridiagonal toeplitz matrix
  int DIM = 10;
  monolish::matrix::COO<double> COO =
      monolish::util::tridiagonal_toeplitz_matrix<double>(DIM, 10.0, -1.0);

  // Edit the matrix as needed //
  // Execute A_COO.sort() after editing the matrix //

  monolish::matrix::CRS<double> A(
      COO); // Create CRS format and convert from COO format

  // Number of eigenvalues needed
  size_t eignum = 2;

  // length eignum
  monolish::vector<double> eigs(eignum, 1.0, 2.0);

  // Dense matrix, the size is eignum * A.get_row() for eigenvectors
  monolish::matrix::Dense<double> eigvecs(eignum, A.get_row(), 0.0);

  // Send to GPU
  monolish::util::send(A);

  // Create LOBPCG solver for  standard eigenvalue
  monolish::standard_eigen::LOBPCG<monolish::matrix::CRS<double>, double>
      solver;

  // if you need generalized eigen solver
  // monolish::generalized_eigen::LOBPCG<monolish::matrix::CRS<T>, T> solver;

  // create jacobi preconditioner
  monolish::equation::Jacobi<monolish::matrix::CRS<double>, double> precond;

  // Set preconditioner to LOBPCG solver
  solver.set_create_precond(precond);
  solver.set_apply_precond(precond);

  // Set solver options
  solver.set_tol(1.0e-12);
  solver.set_maxiter(A.get_row());

  // if you need residual history
  // solver.set_print_rhistory(true);
  // solver.set_rhistory_filename("./a.txt");

  // Solve
  if (monolish::util::solver_check(solver.solve(A, eigs, eigvecs))) {
    return 1;
  };

  // Recv eigenvalues and eigenvectors from GPU
  monolish::util::recv(eigs, eigvecs);

  // Show answer
  std::cout << "== eigenvalue ==" << std::endl;
  eigs.print_all();

  for (size_t i = 0; i < eigs.size(); i++) {
    std::cout << "== eigenvector " << i << "==" << std::endl;
    // view1D is a 1D reference class that have same functions as
    // monolish::vector
    monolish::view1D<monolish::matrix::Dense<double>, double> x(
        eigvecs, i * A.get_col(), (i + 1) * A.get_col());
    x.print_all();
  }

  return 0;
}
