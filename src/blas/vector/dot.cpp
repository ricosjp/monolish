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
	double blas::dot(const vector<double> &x, const vector<double> &y){
		Logger& logger = Logger::get_instance();
		logger.func_in(monolish_func);

		//err
		if( x.size() != y.size()){
			throw std::runtime_error("error vector size is not same");
		}

		double ans = 0;
		const double* xd = x.data();
		const double* yd = y.data();
		const size_t size = x.size();
	
#if USE_GPU
		#pragma acc data pcopyin(xd[0:size], yd[0:size]) 
		#pragma acc host_data use_device(xd, yd)
		{
			ans = cublasDdot(size, xd, 1, yd, 1);
		}
#else
		ans = cblas_ddot(size, xd, 1, yd, 1);
#endif
		logger.func_out();
		return ans;
	}

	// float ///////////////////
	float blas::dot(const vector<float> &x, const vector<float> &y){
		Logger& logger = Logger::get_instance();
		logger.func_in(monolish_func);

		//err
		if( x.size() != y.size()){
			throw std::runtime_error("error vector size is not same");
		}

		float ans = 0;
		const float* xd = x.data();
		const float* yd = y.data();
		const size_t size = x.size();
	
#if USE_GPU
		#pragma acc data pcopyin(xd[0:size], yd[0:size])
		#pragma acc host_data use_device(xd, yd)
		{
			ans = cublasSdot(size, xd, 1, yd, 1);
		}
#else
		ans = cblas_sdot(size, xd, 1, yd, 1);
#endif
		logger.func_out();
		return ans;
	}

	// void ///////////////////
	void blas::dot(const vector<double> &x, const vector<double> &y, double& ans){
		ans = dot(x,y);
	}
	void blas::dot(const vector<float> &x, const vector<float> &y, float& ans){
		ans = dot(x,y);
	}

}
