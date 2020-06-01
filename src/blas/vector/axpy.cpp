#include "../../../include/monolish_blas.hpp"
#include "../../monolish_internal.hpp"

#ifdef USE_GPU
	#include<cublas_v2.h>
#else
	#include<cblas.h>
#endif

namespace monolish{

	// double ///////////////////
	void blas::axpy(const double alpha, const vector<double> &x, vector<double> &y){
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
			cublasHandle_t h;
			check(cublasCreate(&h));
			#pragma acc host_data use_device(xd, yd)
			{
				check(cublasDaxpy(h, size, &alpha, xd, 1, yd, 1));
			}
			cublasDestroy(h);
		#else
			cblas_daxpy(size, alpha, xd, 1, yd, 1);
		#endif
		logger.func_out();
	}

	// float ///////////////////
	void blas::axpy(const float alpha, const vector<float> &x, vector<float> &y){
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
			cublasHandle_t h;
			check(cublasCreate(&h));
			#pragma acc host_data use_device(xd, yd)
			{
				check(cublasSaxpy(h, size, &alpha, xd, 1, yd, 1));
			}
			cublasDestroy(h);
		#else
			cblas_saxpy(size, alpha, xd, 1, yd, 1);
		#endif
		logger.func_out();
	}
}
