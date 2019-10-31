#include<iostream>
#include"../include/monolish_equation.hpp"
#include"../include/common/monolish_common.hpp"

int main(){

	monolish::equation::cg cg_solver;
	monolish::COO_matrix<double> COO("./test.mtx");
	monolish::CRS_matrix<double> A(COO);

	monolish::vector<double> x(A.get_row(), 0.0);
	monolish::vector<double> b(A.get_row(), 1.0);


	cg_solver.set_tol(1.0e-12);
	cg_solver.set_maxiter(A.get_row());
	cg_solver.solve(A, x, b);



	return 0;


}
