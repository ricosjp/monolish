#pragma once
#include"monolish_logger.hpp"
#include"monolish_matrix.hpp"
#include"monolish_vector.hpp"
#include<initializer_list>

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
//send///////////////////

		/**
		 * @brief send data to GPU
		 **/
		template < typename T >
			auto send( T& x )
			{
				x.send();
			}

		/**
		 * @brief send datas to GPU
		 **/
		template < typename T, typename ... Types >
			auto send( T& x, Types& ... args )
			{
				x.send();
				send( args... );
			}

//recv///////////////////
		/**
		 * @brief recv data from GPU
		 **/
		template < typename T >
			auto recv( T& x )
			{
				x.recv();
			}

		/**
		 * @brief recv datas to GPU
		 **/
		template < typename T, typename ... Types >
			auto recv( T& x, Types& ... args )
			{
				x.recv();
				recv( args... );
			}

//device_free///////////////////

		/**
		 * @brief recv data from GPU
		 **/
		template < typename T >
			auto device_free( T& x )
			{
				x.device_free();
			}

		/**
		 * @brief recv datas to GPU
		 **/
		template < typename T, typename ... Types >
			auto device_free( T& x, Types& ... args )
			{
				x.device_free();
				recv( args... );
			}
	}
}
