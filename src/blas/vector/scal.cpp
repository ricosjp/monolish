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
	void blas::scal(const double alpha, vector<double> &x){
		Logger& logger = Logger::get_instance();
		logger.func_in(monolish_func);

		double* xd = x.data();
		size_t size = x.size();
	
#if USE_GPU
		#pragma acc data copy(xd[0:size])
		#pragma acc host_data use_device(xd)
		{
			cublasDscal(size, alpha, xd, 1);
		}
#else
		cblas_dscal(size, alpha, xd, 1);
#endif
		logger.func_out();
	}
}
