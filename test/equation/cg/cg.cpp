#include<iostream>
#include"../include/monolish_equation.hpp"
#include"../include/monolish_blas.hpp"

int main(int argc, char** argv){
	if(argc!=2){
		std::cout << "error $1 is matrix filename" << std::endl;
		return 1;
	}

	//monolish::util::set_log_level(3);
	//monolish::util::set_log_filename("./monolish_test_log.txt");

	char* file = argv[1];
	monolish::matrix::COO<double> COO(file);
	monolish::matrix::CRS<double> A(COO);

	monolish::vector<double> ans(A.get_row(), 1.0);
	monolish::vector<double> b(A.get_row(), 0.0);

	// initial x is rand(0~1)
	monolish::vector<double> x(A.get_row(), 123.0);

	A.send();
	monolish::util::send(x, b, ans);

	// create answer
	monolish::blas::spmv(A, ans, b);

	monolish::equation::CG solver;

	solver.set_tol(1.0e-12);
	solver.set_lib(0);
	solver.set_precon(2);
	solver.set_miniter(5);
	solver.set_maxiter(10);

	solver.set_print_rhistory(true);
	//solver.set_rhistory_filename("./a.txt");

	solver.solve(A, x, b);

	x.recv();

	if(x[0] != 1.0 && x[1] != 1.0 && x[2] != 1.0){
		x.print_all();
		return 1;
	}
	return 0;
}
