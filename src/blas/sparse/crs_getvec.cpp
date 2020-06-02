#include "../../../include/monolish_blas.hpp"
#include "../../monolish_internal.hpp"

namespace monolish{
	namespace matrix{

		//get_diag
		template<typename T>
		void CRS<T>::get_diag(vector<T>& vec){
			Logger& logger = Logger::get_instance();
			logger.func_in(monolish_func);

			//err
			if( get_row() != (size_t)vec.size()){
				throw std::runtime_error("error vector size is not same");
			}

			size_t n = get_row();
			size_t nnz = get_nnz();
			T* vecd = vec.data();

			const T* vald = val.data();
			const int* rowd = row_ptr.data();
			const int* cold = col_ind.data();

			#if USE_GPU // gpu

			#pragma acc data present(vecd[0:n], vald[0:nnz], rowd[0:n+1], cold[0:nnz])
			#pragma acc kernels
			{
				#pragma acc loop independent 
					for(size_t i = 0 ; i < n; i++){
						vecd[i] = 0;
					}
				#pragma acc loop independent 
					for(size_t i = 0 ; i < n; i++){
						for(size_t j = rowd[i] ; j < rowd[i+1]; j++){
							if(i == cold[j]){
								vecd[i] = vald[j];
							}
						}
					}
			}
			#else // cpu

				#pragma omp parallel for 
					for(size_t i = 0 ; i < get_row(); i++)
						vecd[i] = 0;

				#pragma omp parallel for 
					for(size_t i = 0 ; i < n; i++){
						for(int j = rowd[i] ; j < rowd[i+1]; j++){
							if((int)i == cold[j]){
								vecd[i] = vald[j];
							}
						}
					}
			#endif

			logger.func_out();
		}
		template void monolish::matrix::CRS<double>::get_diag(vector<double>& vec);
		template void monolish::matrix::CRS<float>::get_diag(vector<float>& vec);

		//get_row
		template<typename T>
		void CRS<T>::get_row(const size_t r, vector<T>& vec){
			Logger& logger = Logger::get_instance();
			logger.func_in(monolish_func);

			//err
			if( get_row() != (size_t)vec.size()){
				throw std::runtime_error("error vector size is not same");
			}

			size_t n = get_row();
			size_t nnz = get_nnz();
			T* vecd = vec.data();

			const T* vald = val.data();
			const int* rowd = row_ptr.data();
			const int* cold = col_ind.data();

			#if USE_GPU // gpu

			#pragma acc data present(vecd[0:n], vald[0:nnz], rowd[0:n+1], cold[0:nnz])
			#pragma acc kernels
			{
				#pragma acc loop independent 
					for(int i = 0 ; i < n; i++){
						vecd[i] = 0;
					}
				#pragma acc loop independent 
					for(int j = rowd[r] ; j < rowd[r+1]; j++){
						vecd[col_ind[j]] = vald[j];
					}
			}
			#else // cpu

				#pragma omp parallel for 
					for(size_t i = 0 ; i < get_row(); i++){
						vecd[i] = 0;
					}

				#pragma omp parallel for 
					for(int j = rowd[r] ; j < rowd[r+1]; j++){
						vecd[col_ind[j]] = vald[j];
					}
			#endif

			logger.func_out();
		}
		template void monolish::matrix::CRS<double>::get_row(const size_t r, vector<double>& vec);
		template void monolish::matrix::CRS<float>::get_row(const size_t r, vector<float>& vec);

		//get_row
		template<typename T>
		void CRS<T>::get_col(const size_t c, vector<T>& vec){
			Logger& logger = Logger::get_instance();
			logger.func_in(monolish_func);

			//err
			if( get_row() != (size_t)vec.size()){
				throw std::runtime_error("error vector size is not same");
			}

			size_t n = get_row();
			size_t nnz = get_nnz();
			T* vecd = vec.data();

			const T* vald = val.data();
			const int* rowd = row_ptr.data();
			const int* cold = col_ind.data();

			#if USE_GPU // gpu

			#pragma acc data present(vecd[0:n], vald[0:nnz], rowd[0:n+1], cold[0:nnz])
			#pragma acc kernels
			{
				#pragma acc loop independent 
					for(size_t i = 0 ; i < n; i++){
						vecd[i] = 0;
					}
				#pragma acc loop independent 
					for(size_t i = 0 ; i < n; i++){
						for(size_t j = rowd[i] ; j < rowd[i+1]; j++){
							if(c == cold[j]){
								vecd[i] = vald[j];
							}
						}
					}
			}
			#else // cpu

				#pragma omp parallel for 
					for(size_t i = 0 ; i < get_row(); i++){
						vecd[i] = 0;
					}

				#pragma omp parallel for 
					for(size_t i = 0 ; i < n; i++){
						for(int j = rowd[i] ; j < rowd[i+1]; j++){
							if((int)c == cold[j]){
								vecd[i] = vald[j];
							}
						}
					}
			#endif

			logger.func_out();
		}
		template void monolish::matrix::CRS<double>::get_col(const size_t c, vector<double>& vec);
		template void monolish::matrix::CRS<float>::get_col(const size_t c, vector<float>& vec);
	}
}
