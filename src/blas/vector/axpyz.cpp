#include<iostream>
#include<typeinfo>

#include<stdio.h>
#include<stdlib.h>
#include<omp.h>
#include "../../../include/monolish_blas.hpp"

#ifdef USE_GPU
	#include<cublas.h>
#else
	#include<cblas.h>
#endif

namespace monolish{

	// double ///////////////////
	void blas::axpyz(const double alpha, const vector<double> &x, const vector<double> &y, vector<double> &z){
		Logger& logger = Logger::get_instance();
		logger.func_in(monolish_func);

		//err
		if( x.size() != y.size()){
			throw std::runtime_error("error vector size is not same");
		}

		const double* xd = x.data();
		const double* yd = y.data();
		double* zd = z.data();
		size_t size = x.size();
	
#if USE_GPU
		#pragma acc data copyin(xd[0:size], yd[0:size])
		#pragma acc host_data use_device(xd, yd)
		{
		}
		#pragma acc data copyout(yd[0:size])
#else
		#pragma omp parallel for
		for(size_t i = 0; i < x.size(); i++){
			zd[i] = alpha * xd[i] + yd[i];
		}
#endif
 		logger.func_out();
 	}
}
