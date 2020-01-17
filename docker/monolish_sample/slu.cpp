#include<iostream>
#include"monolish_equation.hpp"
#include"monolish_blas.hpp"

int main(int argc, char** argv){


	monolish::equation::LU LU_solver;

	monolish::matrix::COO<double> COO("test.mtx");
	monolish::matrix::CRS<double> A(COO);

	// ans is 1
	monolish::vector<double> ans(A.get_row(), 1.0);
	monolish::vector<double> b(A.get_row(), 0.0);
	monolish::blas::spmv(A, ans, b);

	// initial x is 5.0 (Todo: change rand)
	monolish::vector<double> x(A.get_row(), 5.0);

	LU_solver.solve(A, x, b);

	x.print_all();

	return 0;
}
