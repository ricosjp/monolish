#include "../../../../include/monolish_blas.hpp"
#include "../../../monolish_internal.hpp"

#ifdef USE_GPU
	#include<cublas_v2.h>
#else
	#include<cblas.h>
#endif

namespace monolish{

	// double ///////////////////
	void blas::matvec(const matrix::Dense<double> &A, const vector<double> &x, vector<double> &y){
		Logger& logger = Logger::get_instance();
		logger.func_in(monolish_func);

		//err
		if( A.get_col() != x.size()){
			throw std::runtime_error("error A.col != x.size");
		}

		if( A.get_row() != y.size()){
			throw std::runtime_error("error A.row != y.size");
		}


		const double* xd = x.data();
		double* yd = y.data();
		const double* val = A.val.data();
		const size_t m = A.get_row();
		const size_t n = A.get_col();
        const double alpha = 1.0;
        const double beta = 0.0;
	
 		#if USE_GPU
			cublasHandle_t h;
			check(cublasCreate(&h));
		    #pragma acc data present(xd[0:n], yd[0:m], val[0:m*n])
            #pragma acc host_data use_device(xd, yd, val)
			{
                //cublas is col major
				check(cublasDgemv(h, CUBLAS_OP_T, n, m, &alpha, val, n, xd, 1, &beta, yd, 1));
			}
			cublasDestroy(h);
 		#else
            cblas_dgemv(CblasRowMajor, CblasNoTrans, m, n, alpha, val, n, xd, 1, beta, yd, 1);
 		#endif
		logger.func_out();
	}

	// float ///////////////////
	void blas::matvec(const matrix::Dense<float> &A, const vector<float> &x, vector<float> &y){
		Logger& logger = Logger::get_instance();
		logger.func_in(monolish_func);

		//err
		if( A.get_col() != x.size()){
			throw std::runtime_error("error A.col != x.size");
		}

		if( A.get_row() != y.size()){
			throw std::runtime_error("error A.row != y.size");
		}


		const float* xd = x.data();
		float* yd = y.data();
		const float* val = A.val.data();
		const size_t n = A.get_row();
		const size_t m = A.get_col();
        const float alpha = 1.0;
        const float beta = 0.0;
	
 		#if USE_GPU
			cublasHandle_t h;
			check(cublasCreate(&h));
		    #pragma acc data present(xd[0:m], yd[0:n], val[0:m*n])
            #pragma acc host_data use_device(xd, yd, val)
			{
                //cublas is col major
				check(cublasSgemv(h, CUBLAS_OP_T, m, n, &alpha, val, m, xd, 1, &beta, yd, 1));
			}
			cublasDestroy(h);
 		#else
            cblas_sgemv(CblasRowMajor, CblasNoTrans, n, m, alpha, val, m, xd, 1, beta, yd, 1);
 		#endif
		logger.func_out();
	}

	template<typename T>
	vector<T> matrix::Dense<T>::operator*(vector<T>& vec){
			vector<T> y(get_row()); 
			y.send();

			blas::matvec(*this, vec, y);

			y.nonfree_recv();

			return y;
		}
	template vector<double> matrix::Dense<double>::operator*(vector<double>& vec);
	template vector<float> matrix::Dense<float>::operator*(vector<float>& vec);
}
