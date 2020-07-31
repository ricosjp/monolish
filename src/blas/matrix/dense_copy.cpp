#include "../../../include/monolish_blas.hpp"
#include "../../monolish_internal.hpp"

namespace monolish{
	namespace matrix{

		// copy
		template<typename T>
			Dense<T> Dense<T>::copy(){
				Logger& logger = Logger::get_instance();
				logger.util_in(monolish_func);

				if( get_device_mem_stat() ) { nonfree_recv(); } // gpu copy

				Dense<T> tmp;
				std::copy(val.data(), val.data()+nnz, tmp.val.begin());
				tmp.row = get_row();
				tmp.col = get_col();
				tmp.nnz = get_nnz();
				if( get_device_mem_stat() ) { tmp.send(); } // gpu copy

				logger.util_out();
				return tmp;
			}

		template Dense<double> Dense<double>::copy();
		template Dense<float> Dense<float>::copy();

		// copy monolish Dense
		template<typename T>
			void Dense<T>::operator=(const Dense<T>& mat){
				Logger& logger = Logger::get_instance();
				logger.util_in(monolish_func);

				val.resize(mat.get_nnz());

				row = mat.get_row();
				col = mat.get_col();
				nnz = mat.get_nnz();
				size_t NNZ = nnz;

				// gpu copy and recv
				if( mat.get_device_mem_stat() ) {
					send();
					T* vald = val.data();
					const T* Mvald = mat.val.data();

					#if USE_GPU

					#pragma acc data present(vald[0:nnz])
					#pragma acc parallel
					#pragma acc loop independent 
					for(size_t i = 0 ; i < NNZ; i++){
						vald[i] = Mvald[i];
					}

					nonfree_recv();
					#endif
				}
				else{
					std::copy(mat.val.data(), mat.val.data()+nnz, val.begin());
				}

				logger.util_out();
			}

		template void Dense<double>::operator=(const Dense<double>& mat);
		template void Dense<float>::operator=(const Dense<float>& mat);

		//copy constractor
		template <typename T>
			Dense<T>::Dense(const Dense<T>& mat){
				Logger& logger = Logger::get_instance();
				logger.util_in(monolish_func);

				val.resize(mat.get_nnz());

				row = mat.get_row();
				col = mat.get_col();
				nnz = mat.get_nnz();
				size_t NNZ = nnz;

				// gpu copy and recv
				if( mat.get_device_mem_stat() ) {
					send();
					T* vald = val.data();

					const T* Mvald = mat.val.data();

					#if USE_GPU

					#pragma acc data present(vald[0:nnz])
					#pragma acc parallel
					#pragma acc loop independent 
					for(size_t i = 0 ; i < NNZ; i++){
						vald[i] = Mvald[i];
					}

					nonfree_recv();
					#endif
				}
				else{
					std::copy(mat.val.data(), mat.val.data()+nnz, val.begin());
				}

				logger.util_out();
			}
		template Dense<double>::Dense(const Dense<double>& mat);
		template Dense<float>::Dense(const Dense<float>& mat);
	}
}
