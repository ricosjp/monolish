/**
 * @autor RICOS Co. Ltd.
 * @file monolish_vector.h
 * @brief declare vector class
 * @date 2019
 **/

#pragma once
#include"monolish_matrix.hpp"

namespace monolish{
    template<typename Float> class vector;
    namespace matrix{
        /**
         * @brief Dense Matrix
         */
        template<typename Float>
            class Dense{
                private:

                    size_t row;
                    size_t col;
                    size_t nnz;

                    bool gpu_flag = false; // not impl
                    bool gpu_status = false; // true: sended, false: not send

					void set_row(const size_t N){row = N;};
					void set_col(const size_t M){col = M;};
					void set_nnz(const size_t NZ){nnz = NZ;};

                public:
                    std::vector<Float> val;

                    void convert(COO<Float> &coo);

					size_t size() const{return row;}
					size_t get_row() const{return row;}
					size_t get_col() const{return col;}
					size_t get_nnz() const{return row*col;}

                    Dense(){}
					Dense(const Dense<Float> &mat);

                    Dense(const size_t N, const size_t M){
						set_row(N);
						set_col(M);
						set_nnz(N*M);

						val.resize(nnz);
                    }

                    Dense(const size_t N, const size_t M, const Float* &value){
						set_row(N);
						set_col(M);
						set_nnz(N*M);

						val.resize(nnz);
						std::copy(value, value+nnz, val.begin());
                    }

                    //rand
                    Dense(const size_t N, const size_t M, const Float min, const Float max){
						set_row(N);
						set_col(M);
						set_nnz(N*M);

						val.resize(nnz);

                        std::random_device random;
                        std::mt19937 mt(random());
                        std::uniform_real_distribution<> rand(min,max);

                        #pragma omp parallel for
                        for(size_t i=0; i<val.size(); i++){
                            val[i] = rand(mt);
                        }
                    }

                    
                    Dense(const size_t N, const size_t M, const Float value){
						set_row(N);
						set_col(M);
						set_nnz(N*M);

						val.resize(nnz);

                        #pragma omp parallel for
                        for(size_t i=0; i<val.size(); i++){
                            val[i] = value;
                        }
                    }

                    /**
                     * @brief get element A[i][j] (only CPU)
                     * @param[in] i row
                     * @param[in] j col 
				     * @return A[i][j]
                     **/
                    Float at(size_t i, size_t j){
                        if( get_device_mem_stat() ) {
                            throw std::runtime_error("Error, GPU vector cant use operator[]");
                        }
                        if(get_row() < i){
                            throw std::runtime_error("Error, A.row < i");
                        }
                        if(get_row() < j){
                            throw std::runtime_error("Error, A.col < j");
                        }
                        return val[get_col() * i + j];
                    }

                    /**
                     * @brief insert element, A[i][j] = val (only CPU)
                     * @param[in] i row
                     * @param[in] j col
                     * @param[in] val value
                     **/
                    void insert(size_t i, size_t j, Float val){
                        if( get_device_mem_stat() ) {
                            throw std::runtime_error("Error, GPU vector cant use operator[]");
                        }
                        if(get_row() < i){
                            throw std::runtime_error("Error, A.row < i");
                        }
                        if(get_row() < j){
                            throw std::runtime_error("Error, A.col < j");
                        }
                        val[get_col() * i + j] = val;
                    }

                    void print_all();

                    // communication ///////////////////////////////////////////////////////////////////////////
                    /**
                     * @brief send data to GPU
                     **/
                    void send();

                    /**
                     * @brief recv and free data from GPU
                     **/
                    void recv();

                    /**
                     * @brief recv data from GPU (w/o free)
                     **/
                    void nonfree_recv();

                    /**
                     * @brief free data on GPU
                     **/
                    void device_free();

                    /**
                     * @brief false; // true: sended, false: not send
                     * @return true is sended.
                     * **/
                    bool get_device_mem_stat() const{ return gpu_status; }

                    /**
                     * @brief; free gpu mem.
                     * **/
                    ~ Dense(){
//                         if(get_device_mem_stat()){
//                             device_free();
//                         }
                    }

                    /////////////////////////////////////////////////////////////////////////////
                    void get_diag(vector<Float>& vec);
                    void get_row(const size_t r, vector<Float>& vec);
                    void get_col(const size_t c, vector<Float>& vec);

                    /////////////////////////////////////////////////////////////////////////////

                    /**
                     * @brief matrix copy
                     * @return copied Dense matrix
                     **/
                    Dense copy();

                    /**
                     * @brief copy matrix, It is same as copy()
                     * @return output matrix
                     **/
                    void operator=(const Dense<Float>& mat);

                    //mat - scalar
                    Dense<Float> operator*(const Float value);

                    //mat - vec
                    vector<Float> operator*(vector<Float>& vec);

                    //mat - mat
                    Dense<Float> operator*(const Dense<Float>& B);
            };
    }
}
