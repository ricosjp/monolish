#include "../../include/monolish_equation.hpp"
#include "../../include/monolish_blas.hpp"
#include<iostream>

namespace monolish{

	void equation::cg::monolish_cg(matrix::CRS<double> &A, vector<double> &x, vector<double> &b){
		Logger& logger = Logger::get_instance();
		logger.solver_in(monolish_func);

		for(size_t iter = 0; iter < maxiter; iter++)
		{
			if(precon_num == 1){
				equation::jacobi jacobi;
				jacobi.solve(A,x,b);
			}
			auto ans = blas::dot(x, b);
			blas::spmv(A, b, x); // x = Ab
		}


		logger.solver_out();

	}

	void equation::cg::solve(matrix::CRS<double> &A, vector<double> &x, vector<double> &b){
		Logger& logger = Logger::get_instance();
		logger.solver_in(monolish_func);

		if(lib == 0){
			monolish_cg(A, x, b);
		}

		logger.solver_out();
	}
}
