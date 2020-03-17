#include<iostream>
#include"../include/monolish_equation.hpp"

int main(int argc, char** argv){
	if(argc!=2){
		std::cout << "error $1 is matrix filename" << std::endl;
		return 1;
	}

	monolish::util::set_log_level(2);

	char* file = argv[1];
	monolish::matrix::COO<double> COO(file);
	monolish::matrix::CRS<double> A(COO);

	monolish::vector<double> x(A.get_row(), 0.0);
	monolish::vector<double> b(A.get_row(), 1.0);

	monolish::equation::CG CG_solver;

	CG_solver.set_tol(1.0e-12);
	CG_solver.set_lib(0);
	CG_solver.set_precon(1);
	CG_solver.set_maxiter(A.get_row());

	CG_solver.solve(A, x, b);

	x.print_all();

	return 0;
}
