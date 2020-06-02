#include "../../include/monolish_equation.hpp"
#include "../../include/monolish_blas.hpp"
#include "../monolish_internal.hpp"

namespace monolish{

	////////////////////////////////////////
	// precon none /////////////////////////
	////////////////////////////////////////
	template <typename T>
	void equation::none::precon_create(matrix::CRS<T>& A){
		Logger& logger = Logger::get_instance();
		logger.solver_in(monolish_func);
		logger.solver_out();
	}
	template void equation::none::precon_create(matrix::CRS<double>& A);
	template void equation::none::precon_create(matrix::CRS<float>& A);

	/////

	template<typename T>
	void equation::none::precon_apply(const vector<T>& r, vector<T>& z){
		Logger& logger = Logger::get_instance();
		logger.solver_in(monolish_func);
		z = r;
		logger.solver_out();
	}
	template void equation::none::precon_apply(const vector<double>& r, vector<double>& z);
	template void equation::none::precon_apply(const vector<float>& r, vector<float>& z);

	/////

	template<typename T>
	int equation::none::solve(matrix::CRS<T> &A, vector<T> &x, vector<T> &b){
		Logger& logger = Logger::get_instance();
		logger.solver_in(monolish_func);
		return MONOLISH_SOLVER_SUCCESS;
		logger.solver_out();
	}
	template int equation::none::solve(matrix::CRS<float> &A, vector<float> &x, vector<float> &b);
	template int equation::none::solve(matrix::CRS<double> &A, vector<double> &x, vector<double> &b);

	//////////////////////////////////////////////////////
	// solver set precon /////////////////////////////////
	//////////////////////////////////////////////////////
	void equation::solver::set_precon_create(std::function<void(matrix::CRS<double>&)> f){
		d_precon.precon_create = f;
	}
	void equation::solver::set_precon_create(std::function<void(matrix::CRS<float>&)> f){
		f_precon.precon_create = f;
	}

	/////
	void equation::solver::set_precon_apply(std::function<void(const vector<double>& r, vector<double>& z)> f){
		d_precon.precon_apply = f;
	}
	void equation::solver::set_precon_apply(std::function<void(const vector<float>& r, vector<float>& z)> f){
		f_precon.precon_apply = f;
	}


	//////////////////////////////////////////////////////
	void equation::solver::precon_create(matrix::CRS<double>& A){
		d_precon.precon_create(A);
	}
	void equation::solver::precon_create(matrix::CRS<float>& A){
		f_precon.precon_create(A);
	}

	void equation::solver::precon_apply(const vector<double>& r, vector<double>& z){
		d_precon.precon_apply(r, z);
	}

	void equation::solver::precon_apply(const vector<float>& r, vector<float>& z){
		f_precon.precon_apply(r, z);
	}
}
