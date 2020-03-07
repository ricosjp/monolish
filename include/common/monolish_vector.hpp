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
#include<iterator>

#include<memory>

#if defined USE_MPI
#include<mpi.h>
#endif
//#typedef typename std::allocator_trails::allocator_type::reference reference;


namespace monolish{

	
/**
 * @class vector
 * @brief it likes std::vector, it has flag that moniter changing value...
**/
	template<typename Float>
		class vector{
			private:

			public:
				bool flag = 0; // 1 mean "changed", not impl..
				std::vector<Float> val;

/**
 * @fn vector()
 * @brief constructor
**/
				vector(){}

/**
 * @fn vector(size_t N)
 * @brief initialize size N vector
 * @param (N) vector size
**/
				vector(size_t N){
					val.resize(N);
				}

/**
 * @fn vector(size_t N, Float a)
 * @brief initialize size N vector, value to fill the container
 * @param (N) vector size
 * @param (val) fill Float type value to all elements
**/
				vector(size_t N, Float a){
					val.resize(N, a);
				}

/**
 * @fn vector(std::vector<Float> vec)
 * @brief copy std::vector
 * @param (vec) 
**/

				vector(std::vector<Float>& vec){
					val.resize(vec.size());
					std::copy(vec.begin(), vec.end(), val.begin());
				}

/**
 * @fn vector()
 * @brief copy std::vector
 * @param (vec) 
**/
				vector(const Float* start, const Float* end){
					size_t size = (end - start);
					val.resize(size);
					std::copy(start, end, val.begin());
				}

/////////////////////////////////////////////////////////////////////////////

/**
 * @fn data()
 * @brief returns a direct pointer to the vector
 * @return A pointer to the first element
**/
				Float* data(){
					return val.data();
				}

/**
 * @fn size()
 * @brief get vector size N
 * @return vector size
**/
				size_t size() const{
					return val.size();
				}

/**
 * @fn copy()
 * @brief vector copy
 * @return copied vector
**/
				vector copy(){
					vector<Float> tmp(val.size());
					std::copy(val.begin(), val.end(), tmp.val.begin());
					return tmp;
				}

/**
 * @fn print_all()
 * @brief print all elements to standart I/O
 * @return vector 
**/
				void print_all(){
					for(size_t i = 0; i < val.size(); i++){
						std::cout <<  val[i] << std::endl;
					}
				}

				//extern void add(const vector<Float> &x, const vector<Float> &y);
				void add(const vector<Float> &x, const vector<Float> &y);

				vector operator+(vector<Float>& vec);
				vector operator+=(vector<Float>& vec);
				vector operator+(double value);

				Float& operator [] ( size_t i){
					return val[i];
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

/**
 * @fn random_vector()
 * @brief create random vector
 * @return ramdom vector 
**/
}
