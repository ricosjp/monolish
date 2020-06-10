#include "../../include/monolish_equation.hpp"
#include "../../include/monolish_blas.hpp"
#include "../monolish_internal.hpp"
#include<iostream>
#include<fstream>
#include<string>
#include<iomanip>

namespace monolish{

	template<typename T>
	int equation::CG<T>::monolish_CG(matrix::CRS<T> &A, vector<T> &x, vector<T> &b){
		Logger& logger = Logger::get_instance();
		logger.solver_in(monolish_func);
		std::ostream* pStream;

		vector<T> r(A.size(), 0.0);
		vector<T> p(A.size(), 0.0);
		vector<T> q(A.size(), 0.0);
		vector<T> z(A.size(), 0.0);
		monolish::util::send(r,p,q,z);

		if( A.get_device_mem_stat() == false) { A.send(); }
		if( x.get_device_mem_stat() == false) { x.send(); }
		if( b.get_device_mem_stat() == false) { b.send(); }

		this->precond.precond_create(A);

		//r = b-Ax
		blas::spmv(A, x, q);
		r = b - q;

		//p0 = Mr0
		p = r;
		this->precond.precond_apply(r, z);

		for(size_t iter = 0; iter < this->maxiter; iter++)
		{
			blas::spmv(A,p,q);

			auto tmp = blas::dot(z,r);
			auto alpha = tmp / blas::dot(p,q);

 			blas::axpy(alpha, p, x);

 			blas::axpy(-alpha, q, r);

			this->precond.precond_apply(r, z);
 			auto beta = blas::dot(z,r) / tmp;

 			blas::xpay(beta, r, p);//p = r + beta*p

			T resid = this->get_residual(r);
			if(this->print_rhistory==true){
				*this->rhistory_stream << iter+1 << "\t" << std::scientific << resid << std::endl;
			}

			if( resid < this->tol && this->miniter <= iter+1){
				logger.solver_out();
				return MONOLISH_SOLVER_SUCCESS;
			} 
		}

		logger.solver_out();
		return MONOLISH_SOLVER_MAXITER;
	}
	template int equation::CG<double>::monolish_CG(matrix::CRS<double> &A, vector<double> &x, vector<double> &b);
	template int equation::CG<float>::monolish_CG(matrix::CRS<float> &A, vector<float> &x, vector<float> &b);

    ///

	template<typename T>
	int equation::CG<T>::solve(matrix::CRS<T> &A, vector<T> &x, vector<T> &b){
		Logger& logger = Logger::get_instance();
		logger.solver_in(monolish_func);

		int ret = 0;
		if(this->lib == 0){
			ret = monolish_CG(A, x, b);
		}

		logger.solver_out();
		return ret; // err code
	}
	template int equation::CG<double>::solve(matrix::CRS<double> &A, vector<double> &x, vector<double> &b);
	template int equation::CG<float>::solve(matrix::CRS<float> &A, vector<float> &x, vector<float> &b);
}
