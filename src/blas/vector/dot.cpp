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

#define BENCHMARK
namespace monolish{


	double blas::dot(vector<double> &x, vector<double> &y){
		Logger& logger = Logger::get_instance();
		logger.func_in(monolish_func);

		//err
		if( x.size() != y.size()){
			throw std::runtime_error("error vector size is not same");
		}

		double ans = 0;
		double* xd = x.data();
		double* yd = y.data();
		size_t size = x.size();
	
#if USE_GPU
		#pragma acc data copyin(xd[0:size], yd[0:size])
		#pragma acc host_data use_device(xd, yd)
		{
			ans = cublasDdot(x.size(), xd, 1, yd, 1);
		}
		#pragma acc data copyout(yd[0:size])
#else
		ans = cblas_ddot(x.size(), x.data(), 1, y.data(), 1);
#endif


		logger.func_out();
		return ans;
	}
}

