#include<iostream>
#include"../../test_utils.hpp"
#include"../include/monolish_equation.hpp"
#include"../include/monolish_blas.hpp"

int main(int argc, char** argv){

	if(argc!=3){
		std::cout << "error $1:matrix filename, $2:error check (1/0)" << std::endl;
		return 1;
	}

	//monolish::util::set_log_level(3);
	//monolish::util::set_log_filename("./monolish_test_log.txt");

	char* file = argv[1];
	int check_ans = atoi(argv[2]);

	monolish::matrix::COO<double> COO(file);
	monolish::matrix::CRS<double> A(COO);

	monolish::vector<double> ans(A.get_row(), 1.0);
	monolish::vector<double> b(A.get_row(), 0.0);

	// initial x is rand(0~1)
	monolish::vector<double> x(A.get_row(), 123.0);

	monolish::util::send(A, x, b, ans);

	// create answer
	monolish::blas::spmv(A, ans, b);

	monolish::equation::CG solver;

	solver.set_tol(1.0e-12);
	solver.set_lib(0);
	solver.set_precon(2);
 	solver.set_miniter(5);
 	solver.set_maxiter(10000);

	solver.set_print_rhistory(true);
	//solver.set_rhistory_filename("./a.txt");

	if (monolish::util::solver_check(solver.solve(A, x, b))) {return 1;}

	//std::cout << monolish::util::get_residual_l2(A,x,b) << std::endl;

	ans.recv();
	x.recv();

	if(check_ans == 1){
		if(ans_check<double>(x.data(), ans.data(), x.size(), 1.0e-8) == false){
			x.print_all();
			return 1;
		};
	}
	return 0;
}
