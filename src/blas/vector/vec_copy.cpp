#include "../../../include/monolish_blas.hpp"
#include "../../monolish_internal.hpp"

namespace monolish{

	// copy
	template<typename T>
	vector<T> vector<T>::copy(){
		Logger& logger = Logger::get_instance();
		logger.util_in(monolish_func);

		if( get_device_mem_stat() ) { nonfree_recv(); } // gpu copy

		vector<T> tmp(val.size());
		std::copy(val.begin(), val.end(), tmp.val.begin());
		if( get_device_mem_stat() ) { tmp.send(); } // gpu copy

		logger.util_out();
		return tmp;
	}

	template vector<double> vector<double>::copy();
	template vector<float> vector<float>::copy();

	// copy std vector
	template<typename T>
	void vector<T>::operator=(const std::vector<T>& vec){
		Logger& logger = Logger::get_instance();
		logger.util_in(monolish_func);

		val.resize(vec.size());
		std::copy(vec.begin(), vec.end(), val.begin());

		logger.util_out();
	}

	template void vector<double>::operator=(const std::vector<double>& vec);
	template void vector<float>::operator=(const std::vector<float>& vec);

	// copy monolish vector
	template<typename T>
	void vector<T>::operator=(const vector<T>& vec){
		Logger& logger = Logger::get_instance();
		logger.util_in(monolish_func);

	   	// gpu copy and recv
		if( vec.get_device_mem_stat() ) {

			if( get_device_mem_stat() ) {
				throw std::runtime_error("Error, No GPU memory allocated for the return value (operator=)");
			}
			if( vec.size() != size() ) {
				throw std::runtime_error("error vector size is not same");
			}
			T* vald = val.data();

			const T* vecd = vec.data();
 			size_t size = vec.size();

			#if USE_GPU
				#pragma acc data present(vecd[0:size], vald[0:size])
				#pragma acc kernels
				#pragma acc loop independent 
				for(size_t i = 0 ; i < size; i++){
					vald[i] = vecd[i];
				}
			#endif
	   	}
		else{
			val.resize(vec.size());
			std::copy(vec.val.begin(), vec.val.end(), val.begin());
		}

		logger.util_out();
	}

	template void vector<double>::operator=(const vector<double>& vec);
	template void vector<float>::operator=(const vector<float>& vec);

	//copy constractor
	template <typename T>
	vector<T>::vector(const monolish::vector<T>& vec){
		Logger& logger = Logger::get_instance();
		logger.util_in(monolish_func);

		val.resize(vec.size());

	   	// gpu copy and recv
		if( vec.get_device_mem_stat() ) {
			std::cout<< "1" << std::endl;
			send();
			T* vald = val.data();
			std::cout<< "2" << std::endl;

			const T* vecd = vec.data();
 			size_t size = vec.size();
			std::cout<< "3" << std::endl;

			#if USE_GPU
				#pragma acc data present(vecd[0:size], vald[0:size])
				#pragma acc kernels
				#pragma acc loop independent 
				for(size_t i = 0 ; i < size; i++){
					vald[i] = vecd[i];
				}

		   	nonfree_recv();
			std::cout<< "4" << std::endl;
			#endif
	   	}
		else{
			std::copy(vec.val.begin(), vec.val.end(), val.begin());
		}

		logger.util_out();
	}
	template vector<double>::vector(const vector<double>& vec);
	template vector<float>::vector(const vector<float>& vec);
}
