#include "../../../include/monolish_equation.hpp"
#include "../../../include/monolish_blas.hpp"
#include "../../monolish_internal.hpp"

#ifdef USE_GPU
#include "cusolverSp.h"
#include "cusparse.h"
#endif

namespace monolish{

	int equation::LU::cusolver_LU(matrix::CRS<double> &A, vector<double> &x, vector<double> &b){
		Logger& logger = Logger::get_instance();
		logger.func_in(monolish_func);
		if( 1 ){
			throw std::runtime_error("error sparse LU on GPU does not impl.");
		}

		logger.func_out();
		return 0;

	}
}
