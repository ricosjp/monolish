#include "../../../include/monolish_equation.hpp"
#include "../../monolish_internal.hpp"

namespace monolish{
	int equation::QR::solve(matrix::CRS<double> &A, vector<double> &x, vector<double> &b){
		Logger& logger = Logger::get_instance();
		logger.func_in(monolish_func);

		int ret = -1;

#if USE_GPU // gpu
		if(lib == 1){
			ret = cusolver_QR(A, x, b);
		}
		else{
			logger.func_out();
			throw std::runtime_error("error solver.lib is not 1");
		}
#else
		logger.func_out();
		throw std::runtime_error("error QR on CPU does not impl.");
#endif

		logger.func_out();
		return ret;
	}

}
