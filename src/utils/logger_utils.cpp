#include "../../include/monolish_blas.hpp"

	void monolish::util::set_log_level(size_t Level){
		Logger& logger = Logger::get_instance();
		logger.set_log_level(Level);
	}

 	void monolish::util::set_log_filename(std::string filename){
		Logger& logger = Logger::get_instance();
		logger.set_log_filename(filename);
	}
