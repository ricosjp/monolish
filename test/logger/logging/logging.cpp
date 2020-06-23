#include<iostream>
#include"../include/monolish_equation.hpp"

int main(int argc, char** argv){
	if(argc!=3){
		std::cout << "error $1 log file name" << std::endl;
		std::cout << "error $2 is matrix filename" << std::endl;
		return 1;
	}

	monolish::util::set_log_level(2);
	monolish::util::set_log_filename(argv[1]);

	char* file = argv[2];
	monolish::matrix::COO<double> COO(file);
	monolish::matrix::CRS<double> A(COO);

	monolish::vector<double> x(A.get_row(), 0.0);
	monolish::vector<double> b(A.get_row(), 1.0);

	monolish::util::send(A,x,b);

	monolish::equation::CG<double> solver;
	monolish::equation::none<double> precond;

    solver.set_create_precond(precond);
    solver.set_apply_precond(precond);

 	solver.set_tol(1.0e-12);
 	solver.set_maxiter(A.get_row());

 	solver.solve(A, x, b);

	return 0;
}
