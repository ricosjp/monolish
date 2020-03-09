#pragma once
#include<omp.h>
#include<chrono>
#include<vector>
#include<iostream>
#include<fstream>
#include<string>

#define monolish_func __FUNCTION__

#if defined USE_MPI
#include<mpi.h>
#endif

namespace monolish{
	class Logger 
	{
		private:
			Logger() = default;

			~Logger(){
				if(pStream != &std::cout){
					delete pStream;
				}
			};
			
			std::vector<std::string> calls;
			std::vector<std::chrono::system_clock::time_point> times;
			double stime;
			double etime;
			std::string filename;
			std::ostream* pStream;

		public:

			size_t LogLevel=0;

			Logger(const Logger&) = delete;
			Logger& operator=(const Logger&) = delete;
			Logger(Logger&&) = delete;
			Logger& operator=(Logger&&) = delete;

			static Logger& get_instance()
			{
				static Logger instance;
				return instance;
			}

			void set_log_level(size_t L){
				if( 3 < L){ // loglevel = {0, 1, 2, 3}
					throw std::runtime_error("error bad LogLevel");
				}
				LogLevel=L;
			}

 			void set_log_filename(std::string file){
				if( 0 ){ 
					throw std::runtime_error("error bad filename");
				}
				filename=file;

				//file open
				pStream	= new std::ofstream(filename);
				if(pStream -> fail()){
					delete pStream;
					pStream = &std::cout;
				}
			}

			// for solver (large func.)
			void solver_in(std::string func_name)
			{
				if(LogLevel >= 1){
					if(filename.empty()){
						pStream = &std::cout;
					}

					//init
					calls.push_back(func_name);
					times.push_back(std::chrono::system_clock::now());

					//start
					*pStream << "{" << std::flush;

					//func
					*pStream << "\"solver\" : " << std::flush;
					*pStream << "\"" << std::flush;
					for(int i=0; i < (int)calls.size(); i++)
						*pStream << calls[i] << "/" << std::flush;
					*pStream << "\"" << std::flush;
					*pStream <<  ", " << std::flush;

					// stat
					*pStream << "\"stat\" : \"IN\"" << std::flush;
					
					//end
					*pStream << "}," << std::endl;
				}

			}

			void solver_out()
			{
				if(LogLevel >= 1){
					if(filename.empty()){
						pStream = &std::cout;
					}

					//start
					*pStream << "{" << std::flush;

					//func
					*pStream << "\"solver\" : " << std::flush;
					*pStream << "\"" << std::flush;
					for(int i=0; i < (int)calls.size(); i++)
						*pStream << calls[i] << "/" << std::flush;
					*pStream << "\"" << std::flush;
					*pStream <<  ", " << std::flush;

					//time
					auto end = std::chrono::system_clock::now();
					double sec = std::chrono::duration_cast<std::chrono::nanoseconds>(end - times[(int)times.size()-1]).count()/1.0e+9;
					*pStream << "\"stat\" : \"OUT\", " << std::flush;
					*pStream << "\"time\" : " << sec << std::flush;

					//end
					*pStream << "}," << std::endl;

					calls.pop_back();
					times.pop_back();
				}
			}

			/////////////////////////////////////////////////////////////////////////////
			/////////////////////////////////////////////////////////////////////////////
			/////////////////////////////////////////////////////////////////////////////

			// for blas (small func.)
			void func_in(std::string func_name)
			{
				if(LogLevel >= 2){
					if(filename.empty()){
						pStream = &std::cout;
					}

					calls.push_back(func_name);
					times.push_back(std::chrono::system_clock::now());
				}
			}

			void func_out()
			{
				if(LogLevel >= 2){
					if(filename.empty()){
						pStream = &std::cout;
					}

					//start
					*pStream << "{" << std::flush;

					//func
					*pStream << "\"func\" : " << std::flush;
					*pStream << "\"" << std::flush;
					for(int i=0; i < (int)calls.size(); i++)
						*pStream << calls[i] << "/" << std::flush;
					*pStream << "\"" << std::flush;
					*pStream <<  ", " << std::flush;

					//time
					auto end = std::chrono::system_clock::now();
					double sec = std::chrono::duration_cast<std::chrono::nanoseconds>(end - times[(int)times.size()-1]).count()/1.0e+9;
					*pStream << "\"time\" : " << sec << std::flush;

					//end
					*pStream << "}," << std::endl;

					calls.pop_back();
					times.pop_back();
				}
			}

			// for utils (very small func.)
			void util_in(std::string func_name)
			{
				if(LogLevel >= 3){
					if(filename.empty()){
						pStream = &std::cout;
					}
					calls.push_back(func_name);
					times.push_back(std::chrono::system_clock::now());
				}
			}

			void util_out()
			{
				if(LogLevel >= 3){
					if(filename.empty()){
						pStream = &std::cout;
					}

					//start
					*pStream << "{" << std::flush;

					//func
					*pStream << "\"util\" : " << std::flush;
					*pStream << "\"" << std::flush;
					for(int i=0; i < (int)calls.size(); i++)
						*pStream << calls[i] << "/" << std::flush;
					*pStream << "\"" << std::flush;
					*pStream <<  ", " << std::flush;

					//time
					auto end = std::chrono::system_clock::now();
					double sec = std::chrono::duration_cast<std::chrono::nanoseconds>(end - times[(int)times.size()-1]).count()/1.0e+9;
					*pStream << "\"time\" : " << sec << std::flush;

					//end
					*pStream << "}," << std::endl;

					calls.pop_back();
					times.pop_back();
				}
			}
	};

	//@fn        set_log_level
	//@var(Level) Log Level
	//@brief      0 : none
	//@brief      1 : all
	//@brief      2 : solver
	//@brief      2 : solver, func
	//@brief      3 : solver, func, util
	void set_log_level(size_t Level);

	//@fn        set_output_file
	//@var(filename) log file name (if not set filename, output standard I/O)
 	void set_log_filename(std::string filename);
}
