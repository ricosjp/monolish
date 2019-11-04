#include<iostream>
#include"../include/monolish_equation.hpp"
#include"../include/common/monolish_common.hpp"

int main(){

	monolish::equation::LU LU_solver;
	monolish::matrix::COO<double> COO("./test.mtx");
	monolish::matrix::CRS<double> A(COO);

	monolish::vector<double> x(A.get_row(), 0.0);
	monolish::vector<double> b(A.get_row(), 1.0);


 	LU_solver.solve(A, x, b);

	monolish::equation::cg CG_solver;
	CG_solver.set_maxiter(100);
	CG_solver.solve(A, x, b);

	return 0;


}
