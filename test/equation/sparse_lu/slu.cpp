#include<iostream>
#include"../../test_utils.hpp"
#include"../include/monolish_equation.hpp"
#include"../include/monolish_blas.hpp"

int main(int argc, char** argv){
	printf("error, slu is not impl.");
	return 1;

// 	if(argc!=3){
// 		std::cout << "error $1:matrix filename, $2:error check (1/0)" << std::endl;
// 		return 1;
// 	}
// 
// 	char* file = argv[1];
// 	int check_ans = atoi(argv[2]);
// 
// 	//monolish::util::set_log_level(3);
// 	//monolish::util::set_log_filename("./monolish_test_log.txt");
// 
// 	monolish::equation::LU LU_solver;
// 
// 	monolish::matrix::COO<double> COO(file);
// 	monolish::matrix::CRS<double> A(COO);
// 
// 	// ans is 1
// 	monolish::vector<double> ans(A.get_row(), 1.0);
// 	monolish::vector<double> b(A.get_row(), 0.0);
// 	monolish::blas::matvec(A, ans, b);
// 
// 	// initial x is rand(0~1)
// 	monolish::vector<double> x(A.get_row(), 0.0, 1.0);
// 
// 	// Solver setting
// 	LU_solver.set_reorder(0);
// 	LU_solver.set_tol(1.0e-12);
// 
// 	LU_solver.solve(A, x, b);
// 
// 	auto singularity = LU_solver.get_sigularity();
// 
// 	if(check_ans == 1){
// 		if(ans_check<double>(x.data(), ans.data(), x.size(), 1.0e-8) == false){
// 			return 1;
// 		};
// 	}
// 
// 	x.print_all();
// 
// 	return 0;
}
