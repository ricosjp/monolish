#include "../../../include/monolish_equation.hpp"
#include "../../monolish_internal.hpp"

namespace monolish{
	int equation::LU::solve(matrix::CRS<double> &A, vector<double> &x, vector<double> &b){
		Logger& logger = Logger::get_instance();
		logger.func_in(monolish_func);

		int ret = -1;

#if USE_GPU // gpu
		if(lib == 1){
			ret = cusolver_LU(A, x, b);
		}
		else{
			logger.func_out();
			throw std::runtime_error("error solver.lib is not 1");
		}
#else
		if(lib == 1){
			ret = mumps_LU(A, x, b);
		}
		else{
			logger.func_out();
			throw std::runtime_error("error solver.lib is not 1");
		}
#endif

		logger.func_out();
		return ret;
	}

}
