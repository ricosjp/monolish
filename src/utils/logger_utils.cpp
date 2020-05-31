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
			std::runtime_error("error, maxiter");
			return false;
		case MONOLISH_SOLVER_BREAKDOWN:
			std::runtime_error("error, breakdown");
			return false;
		case MONOLISH_SOLVER_SIZE_ERROR:
			std::runtime_error("error, size error");
			return false;
		default:
			return 0;
	}
}
