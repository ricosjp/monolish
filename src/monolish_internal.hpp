#include<iostream>
#include<omp.h>
#include<openacc.h>

#ifdef USE_GPU
	#include "cuda_runtime.h"
#endif

namespace monolish{

// error check
#ifdef USE_GPU

	auto checkError = [](auto result, auto func, auto file, auto line) {
		if (result) {
			fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
					file, line, static_cast<unsigned int>(result), cudaGetErrorName((cudaError_t)result), func);
			//cudaDeviceReset();
			exit(EXIT_FAILURE);
		}
	};
	#define check(val) checkError((val), #val, __FILE__, __LINE__)
#endif
}
