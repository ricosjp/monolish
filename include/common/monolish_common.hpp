#pragma once
#include"monolish_logger.hpp"
#include"monolish_matrix.hpp"
#include"monolish_vector.hpp"

namespace monolish{
	namespace util{

		/**
		 * @brief      0 : none<br> 1 : all<br> 2 : solver<br>2 : solver, func <br>3 : solver, func, util
		 * @param[in] Level Log level
		 */
		void set_log_level(size_t Level);

		/**
		 * @brief set output logfile name (defailt=standard I/O)
		 * @param[in] filename log file name (if not set filename, output standard I/O)
		 */
		void set_log_filename(std::string filename);

		/**
		 * @brief create random vector
		 * @return ramdom vector 
		 **/
		template<typename T>
			void random_vector(vector<T>& vec, const T min, const T max){
				// rand (0~1)
				std::random_device random;
				std::mt19937 mt(random());
				std::uniform_real_distribution<> rand(min,max);

				for(size_t i=0; i<vec.size(); i++){
					vec[i] = rand(mt);
				}
			}
	}
}
