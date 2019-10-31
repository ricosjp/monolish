#include<iostream>
#include<typeinfo>

#include<stdio.h>
#include<stdlib.h>
#include<omp.h>

#if USE_GPU
	#include<cublas.h>
#else
	#include<cblas.h>
#endif
#include "../../../include/monolish_blas.hpp"

#define BENCHMARK
namespace monolish{

	double blas::dot(vector<double> &x, vector<double> &y){
		Logger& logger = Logger::get_instance();
		logger.func_in(func);

		//err
		if( x.size() != y.size()){
			throw std::runtime_error("error vector size is not same");

		}

#if USE_GPU
#else
		double ans = cblas_ddot(x.size(), x.data(), 1, y.data(), 1);
#endif


		logger.func_out();
		return ans;
	}

}

