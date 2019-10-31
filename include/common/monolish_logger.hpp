#pragma once
#include<omp.h>
#include<chrono>
#include<vector>
#include<iostream>
#include<string>
#define func __FUNCTION__

#if defined USE_MPI
#include<mpi.h>
#endif

namespace monolish{
	class Logger 
	{
		private:
			Logger() = default;
			~Logger() = default;
			std::vector<std::string> calls;
			std::vector<std::chrono::system_clock::time_point> times;

		public:
			double stime;
			double etime;

			Logger(const Logger&) = delete;
			Logger& operator=(const Logger&) = delete;
			Logger(Logger&&) = delete;
			Logger& operator=(Logger&&) = delete;

			static Logger& get_instance()
			{
				static Logger instance;
				return instance;
			}

			// for solver (large func.)
			void solver_in(
					std::string func_name,
					double tol,
					int maxiter
					)
			{
				//init
				calls.push_back(func_name);
				times.push_back(std::chrono::system_clock::now());

				//func
				std::cout << "\"func\" : " << std::flush;
				std::cout << "\"" << std::flush;
				for(int i=0; i < (int)calls.size(); i++)
					std::cout << calls[i] << "/" << std::flush;
				std::cout << "\"" << std::flush;
				std::cout <<  ", " << std::flush;

				// stat
				std::cout << "\"stat\" : \"IN\"" << std::flush;
				std::cout <<  ", " << std::flush;

				//tol
				std::cout << "\"tol\" : " << tol << std::flush;
				std::cout <<  ", " << std::flush;

				//maxiter
				std::cout << "\"maxiter\" : " << maxiter << std::endl;

			}

			void solver_out()
			{
				//func
				std::cout << "\"func\" : " << std::flush;
				std::cout << "\"" << std::flush;
				for(int i=0; i < (int)calls.size(); i++)
					std::cout << calls[i] << "/" << std::flush;
				std::cout << "\"" << std::flush;
				std::cout <<  ", " << std::flush;

				//time
				auto end = std::chrono::system_clock::now();
				double sec = std::chrono::duration_cast<std::chrono::nanoseconds>(end - times[(int)times.size()-1]).count()/1.0e+9;
				std::cout << "\"time\" : " << sec << std::endl;

				calls.pop_back();
				times.pop_back();
			}

			// for func (small func.)
			void func_in(std::string func_name)
			{
				calls.push_back(func_name);
				times.push_back(std::chrono::system_clock::now());
			}

			void func_out()
			{
				//func
				std::cout << "\"func\" : " << std::flush;
				std::cout << "\"" << std::flush;
				for(int i=0; i < (int)calls.size(); i++)
					std::cout << calls[i] << "/" << std::flush;
				std::cout << "\"" << std::flush;
				std::cout <<  ", " << std::flush;

				//time
				auto end = std::chrono::system_clock::now();
				double sec = std::chrono::duration_cast<std::chrono::nanoseconds>(end - times[(int)times.size()-1]).count()/1.0e+9;
				std::cout << "\"time\" : " << sec << std::endl;

				calls.pop_back();
				times.pop_back();
			}

	};
}
