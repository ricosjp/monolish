#include "../../include/monolish_equation.hpp"
#include "../../include/monolish_blas.hpp"
#include<iostream>

namespace monolish{

	void equation::cg::monolish_cg(vector<double> &x, vector<double> b){
		Logger& logger = Logger::get_instance();
		logger.solver_in(func, tol, maxiter);


		std::cout << "I am cg" << std::endl;
		blas::dot(x, b);
		blas::dot(x, b);
		blas::dot(x, b);
		blas::dot(x, b);

		logger.solver_out();

	}

	void equation::cg::solve(vector<double> &x, vector<double> b){
		Logger& logger = Logger::get_instance();
		logger.solver_in(func, tol, maxiter);

		if(optionA == 1){
			monolish_cg(x, b);
		}

		logger.solver_out();
	}

}
