/**
 * @autor RICOS Co. Ltd.
 * @file monolish_vector.h
 * @brief declare vector class
 * @date 2019
**/

#pragma once
#include<omp.h>
#include<vector>
#include<iostream>
#include<string>
#include<exception>
#include<stdexcept>

#if defined USE_MPI
#include<mpi.h>
#endif


namespace monolish{

	
/**
 * @class monolish::vector<typename Float>
 * @brief it likes std::vector(?), it has flag that moniter changing value...
**/
	template<typename Float>
		class vector{
			private:

			public:
				bool flag = 0; // 1 mean "changed", not impl..
				std::vector<Float> val;

				vector(){}

				vector(std::string::size_type N){
					val.resize(N);
				}
				vector(std::string::size_type N, Float a){
					val.resize(N, a);
				}

				Float* data(){
					return val.data();
				}

				std::string::size_type size(){
					return val.size();
				}

				void print_all(){
					for(int i = 0; i < val.size(); i++){
						std::cout <<  val[i] << std::endl;
					}
				}

// 				// need "ref operator[]" 
// 				Float at(size_type n){
// 					return val[n];
// 				}
//
// 				void insert(size_type n, Float a){
// 					val[n] = a;
// 				}
		};
}
