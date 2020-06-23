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
	void blas::xpay(const double alpha, const vector<double> &x, vector<double> &y){
		Logger& logger = Logger::get_instance();
		logger.func_in(monolish_func);

		//err
		if( x.size() != y.size()){
			throw std::runtime_error("error vector size is not same");
		}

		const double* xd = x.data();
		double* yd = y.data();
		size_t size = x.size();
	
		#if USE_GPU
			#pragma acc data present(xd[0:size],yd[0:size])
			#pragma acc parallel
            #pragma acc loop independent 
			for(size_t i = 0 ; i < size; i++){
				yd[i] = xd[i] + alpha * yd[i];
			}
		#else
			#pragma omp parallel for
			for(size_t i = 0; i < size; i++){
				yd[i] = xd[i] + alpha * yd[i];
			}
		#endif
 		logger.func_out();
 	}

	// float ///////////////////
	void blas::xpay(const float alpha, const vector<float> &x, vector<float> &y){
		Logger& logger = Logger::get_instance();
		logger.func_in(monolish_func);

		//err
		if( x.size() != y.size()){
			throw std::runtime_error("error vector size is not same");
		}

		const float* xd = x.data();
		float* yd = y.data();
		size_t size = x.size();
	
		#if USE_GPU
			#pragma acc data present(xd[0:size],yd[0:size])
			#pragma acc parallel
			#pragma acc loop independent 
			for(size_t i = 0 ; i < size; i++){
				yd[i] = xd[i] + alpha * yd[i];
			}
		#else
			#pragma omp parallel for
			for(size_t i = 0; i < size; i++){
				yd[i] = xd[i] + alpha * yd[i];
			}
		#endif
 		logger.func_out();
 	}
}
