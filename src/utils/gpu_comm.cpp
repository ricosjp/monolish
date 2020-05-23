#include<iostream>
#include<typeinfo>

#include<stdio.h>
#include<stdlib.h>
#include<omp.h>
#include "../../include/monolish_blas.hpp"

namespace monolish{
// vec ///////////////////////////////////////

	template<typename T>
	void vector<T>::send(){
		Logger& logger = Logger::get_instance();
		logger.func_in(monolish_func);


#if USE_GPU
		T* d = val.data();
		size_t N = val.size();
		#pragma acc data copyin(d[0:N])
		{}
#endif 
	 	logger.func_out();
	}


	template<typename T>
	void vector<T>::recv(){
		Logger& logger = Logger::get_instance();
		logger.func_in(monolish_func);

#if USE_GPU
		T* d = val.data();
		size_t N = val.size();
		#pragma acc data copyout(d[0:N])
		{}
#endif 
	 	logger.func_out();
	}

	template void vector<float>::send();
	template void vector<double>::send();

	template void vector<float>::recv();
	template void vector<double>::recv();

// mat ///////////////////////////////////
// util ///////////////////////////////////
}
