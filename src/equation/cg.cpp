#include "../../include/monolish_equation.hpp"
#include "../../include/monolish_blas.hpp"
#include "../monolish_internal.hpp"
#include<iostream>
#include<fstream>
#include<string>

namespace monolish{

	int equation::CG::monolish_CG(matrix::CRS<double> &A, vector<double> &x, vector<double> &b){
		Logger& logger = Logger::get_instance();
		logger.solver_in(monolish_func);
		std::ostream* pStream;

		vector<double> r(A.size(), 0.0);
		vector<double> p(A.size(), 0.0);
		vector<double> q(A.size(), 0.0);
		monolish::util::send(r,p,q);

		if(A.get_device_mem_stat()) { A.send(); }
		if(x.get_device_mem_stat()) { x.send(); }
		if(b.get_device_mem_stat()) { b.send(); }

		//r = b-Ax
		blas::spmv(A, x, q);
		r = b - q;

		//p0 = r0
		p = r;

		for(size_t iter = 0; iter < maxiter; iter++)
		{
			blas::spmv(A,p,q);

			auto tmp = blas::dot(r,r);
			auto alpha = tmp / blas::dot(p,q);

 			blas::axpy(alpha, p, x);

 			blas::axpy(-alpha, q, r);

 			auto beta = blas::dot(r,r) / tmp;

 			blas::xpay(beta, r, p);//x = ay+x

			double resid = get_residual(r);
			if(print_rhistory==true){
				*rhistory_stream << iter+1 << "\t" << resid << std::endl;
			}

			if( resid < tol && miniter <= iter+1){
				return 0;
			} // err code (0:sucess)
		}


		logger.solver_out();
		return -1; // err code(-1:max iter)

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
