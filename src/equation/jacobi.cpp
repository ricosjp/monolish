#include "../../include/monolish_equation.hpp"
#include "../../include/monolish_blas.hpp"
#include "../monolish_internal.hpp"
 
namespace monolish{
 
	void equation::Jacobi::precon_create(matrix::CRS<double>& A){
		Logger& logger = Logger::get_instance();
		logger.solver_in(monolish_func);

		auto precon = d_precon;

		if(precon.M.get_device_mem_stat()){
			precon.M.send();
		}

		A.get_diag(precon.M);

		logger.solver_out();
	}

	void equation::Jacobi::precon_create(matrix::CRS<float>& A){
		Logger& logger = Logger::get_instance();
		logger.solver_in(monolish_func);

		auto precon = f_precon;

		if(precon.M.get_device_mem_stat()){
			precon.M.send();
		}

		A.get_diag(precon.M);

		logger.solver_out();
	}


/////

	void equation::Jacobi::precon_apply(const vector<double>& r, vector<double>& z){
		Logger& logger = Logger::get_instance();
		logger.solver_in(monolish_func);

		auto precon = d_precon;

		z = precon.M * r; // x = Ab

		logger.solver_out();
	}

	void equation::Jacobi::precon_apply(const vector<float>& r, vector<float>& z){
		Logger& logger = Logger::get_instance();
		logger.solver_in(monolish_func);

		auto precon = f_precon;

		z = precon.M * r; // x = Ab

		logger.solver_out();
	}


// /////
// 
// template<typename T>
// int equation::Jacobi::solve(matrix::CRS<T> &A, vector<T> &x, vector<T> &b){
// 	Logger& logger = Logger::get_instance();
// 	logger.solver_in(monolish_func);
// 
// 
// 	logger.solver_out();
// 	return MONOLISH_SOLVER_SUCCESS;
// }
// template int equation::Jacobi::solve(matrix::CRS<float> &A, vector<float> &x, vector<float> &b);
// template int equation::Jacobi::solve(matrix::CRS<double> &A, vector<double> &x, vector<double> &b);
}
