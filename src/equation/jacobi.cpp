#include "../../include/monolish_equation.hpp"
#include "../../include/monolish_blas.hpp"
#include<iostream>

namespace monolish{

	//jacobi solver
	int equation::jacobi::monolish_jacobi(matrix::CRS<double> &A, vector<double> &x, vector<double> &b){
		Logger& logger = Logger::get_instance();
		logger.solver_in(monolish_func);

		x = A.get_diag();
		blas::spmv(A, b, x); // x = Ab

		logger.solver_out();
		return 0;
	}
	// jacobi solver
	int equation::jacobi::solve(matrix::CRS<double> &A, vector<double> &x, vector<double> &b){
		int ret = monolish_jacobi(A, x, b);
		return ret;
	}

	int equation::jacobi::Pinit(matrix::CRS<double> &A, vector<double> &x, vector<double> &b){
		return 0;
	}

	int equation::jacobi::Papply(matrix::CRS<double> &A, vector<double> &x, vector<double> &b){
		return 0;
	}
}
