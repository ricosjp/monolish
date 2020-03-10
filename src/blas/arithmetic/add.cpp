#include<iostream>
#include<typeinfo>

#include<stdio.h>
#include<stdlib.h>
#include<omp.h>
#include "../../../include/monolish_blas.hpp"

#define BENCHMARK
namespace monolish{
	/////////////////////////////////////////////////
	// vec - scalar
	/////////////////////////////////////////////////
	template<>
	vector<double> vector<double>::operator+(const double value){
		Logger& logger = Logger::get_instance();
		logger.func_in(monolish_func);

		vector<double> ans(val.size());

		double* vald = val.data();
		double* ansd = ans.data();
 		size_t size = val.size();

		#if USE_GPU
		#pragma acc data pcopyin(vald[0:size]) copyout(ansd[0:size])
		{
			#pragma acc kernels
			{
				#pragma acc loop independent 
				for(size_t i = 0 ; i < size; i++){
					ansd[i] = vald[i] + value;
				}
			}
		}
		#else
		#pragma omp parallel for
		for(size_t i = 0; i < size; i++){
			ansd[i] = vald[i] + value;
		}
		#endif

	 	logger.func_out();
		return ans;
	}

	template<>
	void vector<double>::operator+=(const double value){
		Logger& logger = Logger::get_instance();
		logger.func_in(monolish_func);

		vector<double> ans(val.size());

		double* vald = val.data();
		double* ansd = ans.data();
 		size_t size = val.size();

		#if USE_GPU
		#pragma acc data copy(vald[0:size])
		{
			#pragma acc kernels
			{
				#pragma acc loop independent 
				for(size_t i = 0 ; i < size; i++){
					vald[i] += value;
				}
			}
		}
		#else
		#pragma omp parallel for
		for(size_t i = 0; i < size; i++){
			vald[i] += value;
		}
		#endif

	 	logger.func_out();
	}

	/////////////////////////////////////////////////
	// vec - vec
	/////////////////////////////////////////////////

	template<>
	vector<double> vector<double>::operator+(const vector<double>& vec){
		Logger& logger = Logger::get_instance();
		logger.func_in(monolish_func);

		vector<double> ans(vec.size());

		const double* vecd = vec.data();
		double* vald = val.data();
		double* ansd = ans.data();
 		size_t size = vec.size();

		#if USE_GPU
		#pragma acc data pcopyin(vecd[0:size], vald[0:size]) copyout(ansd[0:size])
		{
			#pragma acc kernels
			{
				#pragma acc loop independent 
				for(size_t i = 0 ; i < size; i++){
					ansd[i] = vecd[i] + vald[i];
				}
			}
		}
		#else
		#pragma omp parallel for
		for(size_t i = 0; i < size; i++){
			ansd[i] = vecd[i] + vald[i];
		}
		#endif

	 	logger.func_out();
		return ans;
	}

	template<>
	void vector<double>::operator+=(const vector<double>& vec){
		Logger& logger = Logger::get_instance();
		logger.func_in(monolish_func);

		const double* vecd = vec.data();
		double* vald = val.data();
 		size_t size = vec.size();

		#if USE_GPU
		#pragma acc data copy(vald[0:size]) pcopyin(vecd[0:size])
		{
			#pragma acc kernels
			{
				#pragma acc loop independent 
				for(size_t i = 0 ; i < size; i++){
					vald[i] += vecd[i];
				}
			}
		}
		#else
		#pragma omp parallel for
		for(size_t i = 0; i < size; i++){
			vald[i] += vecd[i];
		}
		#endif

	 	logger.func_out();
	}
}
