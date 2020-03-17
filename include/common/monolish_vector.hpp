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
#include<fstream>
#include<string>
#include<exception>
#include<stdexcept>
#include<iterator>
#include<random>

#include<memory>

#if defined USE_MPI
#include<mpi.h>
#endif
//#typedef typename std::allocator_trails::allocator_type::reference reference;


namespace monolish{

	/**
	 * @brief std::vector-like vector class
	 */
	template<typename Float>
		class vector{
			private:

			public:
				bool flag = 0; // 1 mean "changed", not impl..
				std::vector<Float> val;

				vector(){}

				/**
				 * @brief allocate size N vector memory space
				 * @param[in] N vector length
				 **/
				vector(const size_t N){
					val.resize(N);
				}

				/**
				 * @brief initialize size N vector, value to fill the container
				 * @param[in] N vector length
				 * @param[in] value fill Float type value to all elements
				 **/
				vector(const size_t N, const Float value){
					val.resize(N, value);
				}

				/**
				 * @brief copy std::vector
				 * @param[in] vec input std::vector
				 **/

				vector(const std::vector<Float>& vec){
					val.resize(vec.size());
					std::copy(vec.begin(), vec.end(), val.begin());
				}

				/**
				 * @brief copy from pointer
				 * @param[in] start start pointer
				 * @param[in] end  end pointer
				 **/
				vector(const Float* start, const Float* end){
					size_t size = (end - start);
					val.resize(size);
					std::copy(start, end, val.begin());
				}

				/**
				 * @brief create N length rand(min~max) vector
				 * @param[in] N  vector length
				 * @param[in] min rand min
				 * @param[in] max rand max
				 **/
				vector(const size_t N, const Float min, const Float max){
					val.resize(N);
					std::random_device random;
					std::mt19937 mt(random());
					std::uniform_real_distribution<> rand(min,max);

					for(size_t i=0; i<val.size(); i++){
						val[i] = rand(mt);
					}
				}


				/////////////////////////////////////////////////////////////////////////////

				/**
				 * @brief returns a direct pointer to the vector
				 * @return A pointer to the first element
				 **/
				Float* data(){
					return val.data();
				}

				/**
				 * @brief returns a direct pointer to the vector
				 * @return A const pointer to the first element
				 **/
				const Float* data() const{
					return val.data();
				}


				/**
				 * @brief get vector size N
				 * @return vector size
				 **/
				size_t size() const{
					return val.size();
				}

				/**
				 * @brief vector copy
				 * @return copied vector
				 **/
				vector copy(){
					vector<Float> tmp(val.size());
					std::copy(val.begin(), val.end(), tmp.val.begin());
					return tmp;
				}

				/**
				 * @brief print all elements to standart I/O
				 **/
				void print_all(){
					for(size_t i = 0; i < val.size(); i++){
						std::cout <<  val[i] << std::endl;
					}
				}

				/**
				 * @brief print all elements to file
				 * @param[in] filename output filename
				 **/
				void print_all(std::string filename){

					std::ofstream ofs(filename);
					if(!ofs){
						throw std::runtime_error("error file cant open");
					}
					for(size_t i = 0; i < val.size(); i++){
						ofs << val[i] << std::endl;
					}
				}

				/**
				 * @brief copy vector, It is same as copy()
				 * @param[in] filename source vector
				 * @return output vector
				 **/
				void operator=(const vector<Float>& vec){
					val.resize(vec.size());
					std::copy(vec.val.begin(), vec.val.end(), val.begin());
				}
				//vec - scalar
				vector<Float> operator+(const Float value);
				void operator+=(const Float value);

				vector<Float> operator-(const Float value);
				void operator-=(const Float value);

				vector<Float> operator*(const Float value);
				void operator*=(const Float value);

				vector<Float> operator/(const Float value);
				void operator/=(const Float value);

				//vec - vec
				vector<Float> operator+(const vector<Float>& vec);
				void operator+=(const vector<Float>& vec);

				vector<Float> operator-(const vector<Float>& vec);
				void operator-=(const vector<Float>& vec);

				vector<Float> operator*(const vector<Float>& vec);
				void operator*=(const vector<Float>& vec);

				vector<Float> operator/(const vector<Float>& vec);
				void operator/=(const vector<Float>& vec);

				Float& operator [] ( size_t i){
					return val[i];
				}
		};
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
