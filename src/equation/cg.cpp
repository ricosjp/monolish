#include "../../include/monolish_equation.hpp"
#include "../../include/monolish_blas.hpp"
#include<iostream>

namespace monolish{

	int equation::CG::monolish_CG(matrix::CRS<double> &A, vector<double> &x, vector<double> &b){
		Logger& logger = Logger::get_instance();
		logger.solver_in(monolish_func);

		for(size_t iter = 0; iter < maxiter; iter++)
		{
			if(precon_num == 1){
				equation::Jacobi Jacobi;
				int Pret = Jacobi.solve(A, x, b);
			}
			auto ans = blas::dot(x, b);
			blas::spmv(A, b, x); // x = Ab

			if(ans == 0.0 && miniter < iter){return 1;} // err code test
		}


		logger.solver_out();
		return 0; // err code

	}

	int equation::CG::solve(matrix::CRS<double> &A, vector<double> &x, vector<double> &b){
		Logger& logger = Logger::get_instance();
		logger.solver_in(monolish_func);

		int ret=-1;
		if(lib == 0){
			ret = monolish_CG(A, x, b);
		}

		logger.solver_out();
		return ret; // err code
	}
}
