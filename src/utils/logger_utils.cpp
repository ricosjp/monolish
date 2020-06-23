#include "../../include/monolish_blas.hpp"

void monolish::util::set_log_level(size_t Level){
	Logger& logger = Logger::get_instance();
	logger.set_log_level(Level);
}

void monolish::util::set_log_filename(std::string filename){
	Logger& logger = Logger::get_instance();
	logger.set_log_filename(filename);
}

bool monolish::util::solver_check( const int err){
	switch(err){
		case MONOLISH_SOLVER_SUCCESS:
			return 0;
		case MONOLISH_SOLVER_MAXITER:
			std::runtime_error("error, maxiter\n");
			return false;
		case MONOLISH_SOLVER_BREAKDOWN:
			std::runtime_error("error, breakdown\n");
			return false;
		case MONOLISH_SOLVER_SIZE_ERROR:
			std::runtime_error("error, size error\n");
			return false;
		case MONOLISH_SOLVER_NOT_IMPL:
			std::runtime_error("error, this solver is not impl.\n");
			return false;
		default:
			return 0;
	}
}
