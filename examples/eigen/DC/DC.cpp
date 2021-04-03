#include<iostream>
#include<monolish_equation.hpp>
#include<monolish_eigen.hpp>

int main(){

  // Output log if you need
  // monolish::util::set_log_level(3);
  // monolish::util::set_log_filename("./monolish_test_log.txt");

  // create tridiagonal toeplitz matrix
  int DIM = 10;
  monolish::matrix::COO<double> COO =
      monolish::util::tridiagonal_toeplitz_matrix<double>(DIM, 10.0, -1.0);

  // Convert from COO
  monolish::matrix::Dense<double> A(COO);
  monolish::vector<double> lambda(DIM);

  // divide and conquer class
  monolish::eigen::DC<monolish::matrix::Dense<double>, double> solver;

  // Solve
  solver.solve(A, lambda);

  lambda.print_all();

  return 0;
}
