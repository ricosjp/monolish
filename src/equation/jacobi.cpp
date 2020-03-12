#include "../../include/monolish_equation.hpp"
#include "../../include/monolish_blas.hpp"
#include<iostream>

namespace monolish{

	void equation::jacobi::solve(matrix::CRS<double> &A, vector<double> &x, vector<double> &b){
		Logger& logger = Logger::get_instance();
		logger.solver_in(monolish_func);

		x = A.get_diag();
		blas::spmv(A, b, x); // x = Ab


		logger.solver_out();
	}
}
