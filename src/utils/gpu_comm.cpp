#include "../../include/monolish_blas.hpp"
#include "../monolish_internal.hpp"

namespace monolish{
// vec ///////////////////////////////////////

	//send
	template<typename T>
	void vector<T>::send(){
		Logger& logger = Logger::get_instance();
		logger.func_in(monolish_func);

#if USE_GPU
		T* d = val.data();
		size_t N = val.size();
		#pragma acc enter data copyin(d[0:N])

		gpu_status=true;
#endif 
	 	logger.func_out();
	}


	//recv
	template<typename T>
	void vector<T>::recv(){
		Logger& logger = Logger::get_instance();
		logger.func_in(monolish_func);

#if USE_GPU
		T* d = val.data();
		size_t N = val.size();
		#pragma acc update host(d[0:N])
#endif 
	 	logger.func_out();
	}

	//device_free
	template<typename T>
	void vector<T>::device_free(){
		Logger& logger = Logger::get_instance();
		logger.func_in(monolish_func);

#if USE_GPU
		T* d = val.data();
		size_t N = val.size();
		#pragma acc exit data delete(d[0:N])
		gpu_status=false;
#endif 
	 	logger.func_out();
	}

	template void vector<float>::send();
	template void vector<double>::send();

	template void vector<float>::recv();
	template void vector<double>::recv();

	template void vector<float>::device_free();
	template void vector<double>::device_free();

// mat ///////////////////////////////////
// util ///////////////////////////////////
}
