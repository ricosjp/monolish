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

	monolish::equation::CG cg_solver;
	char* file = argv[2];
	monolish::matrix::COO<double> COO(file);
	monolish::matrix::CRS<double> A(COO);

	monolish::vector<double> x(A.get_row(), 0.0);
	monolish::vector<double> b(A.get_row(), 1.0);

	monolish::util::send(A,x,b);

 	cg_solver.set_tol(1.0e-12);
 	cg_solver.set_maxiter(A.get_row());

 	cg_solver.solve(A, x, b);

 	//cg_solver.set_precon(1); //jacobi modoki
 	cg_solver.solve(A, x, b);

	return 0;
}
