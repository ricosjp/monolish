#include "../../include/common/monolish_matrix.hpp" 
#include "../../include/common/monolish_logger.hpp" 
#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

//todo: kill cerr 

namespace monolish{
	namespace matrix{

		template<typename T>
			void COO<T>::input_mm(const char* filename){
				Logger& logger = Logger::get_instance();
				logger.util_in(monolish_func);

				std::string banner, buf;
				std::string mm, mat, fmt, dtype, dstruct;

				//file open
				std::ifstream ifs(filename);
				if (!ifs) {
					std::cerr << "Matrix.input: cannot open file " << filename << std::endl;
					std::exit(1);
				}

				//check Matrix Market bannner
				getline(ifs, banner);
				std::istringstream bn(banner);
				bn >> mm >> mat >> fmt >> dtype >> dstruct;

				if (mm != std::string(MM_BANNER)) {
					std::cerr << "Matrix.input: This matrix is not MM format:" << mm << std::endl;
					exit(-1);
				}
				if (mat != std::string(MM_MAT)) {
					std::cerr << "Matrix.input: This matrix is not matrix type:" << mat << std::endl;
					exit(-1);
				}
				if (fmt != std::string(MM_FMT)) {
					std::cerr << "Matrix.input: This matrix is not coodinate format:" << fmt << std::endl;
					exit(-1);
				}
				if (dtype != std::string(MM_TYPE_REAL)) {
					std::cerr << "Matrix.input: This matrix is not real:" << dtype << std::endl;
					exit(-1);
				}
				if (dstruct != std::string(MM_TYPE_GENERAL)) {
					std::cerr << "Matrix.input: This matrix is not general:" << dstruct << std::endl;
					exit(-1);
				}

				//skip %
				do {
					getline(ifs, buf);
				} while (buf[0] == '%');

				//check size
				size_t rowNN, colNN, NNZ;

				std::istringstream data(buf);
				data >> rowNN >> colNN >> NNZ;

				//symmetric check!
				if (colNN != rowNN) {
					std::cerr << "Matrix.input: Matrix is not square" << std::endl;
					exit(-1);
				}
				if (colNN <= 0 || NNZ < 0) {
					std::cerr << "Matrix.input: Matrix size should be positive" << std::endl;
					exit(-1);
				}

				rowN = rowNN;
				colN = rowN;
				nnz = NNZ;

				//allocate
				row_index.resize(nnz, 0.0);
				col_index.resize(nnz, 0.0);
				val.resize(nnz, 0.0);

				//set values
				for(size_t i = 0; i < nnz; i++){
					size_t ix, jx;
					T value;

					getline(ifs, buf);
					std::istringstream data(buf);
					data >> ix >> jx >> value;

					row_index[i] = ix-1;
					col_index[i] = jx-1;
					val[i]		 = value;
				}
				logger.util_out();
			}

		template void COO<double>::input_mm(const char* filename);
		template void COO<float>::input_mm(const char* filename);

		template<typename T>
			void COO<T>::print_all(){
				Logger& logger = Logger::get_instance();
				logger.util_in(monolish_func);
                                std::cout << std::scientific;
                                std::cout << std::setprecision(std::numeric_limits<T>::max_digits10);

                                std::cout << (MM_BANNER " " MM_MAT " " MM_FMT " " MM_TYPE_REAL " " MM_TYPE_GENERAL) << std::endl;
                                std::cout << rowN << " " << colN << " " << nnz << std::endl;

				for(size_t i=0; i<nnz; i++){
                                    std::cout << row_index[i]+1 << " " << col_index[i]+1 << " " << val[i] << std::endl;
				}
				logger.util_out();
			}
		template void COO<double>::print_all();
		template void COO<float>::print_all();

		template<typename T>
			void COO<T>::print_all(std::string filename){
				Logger& logger = Logger::get_instance();
				logger.util_in(monolish_func);
				std::ofstream out(filename);
				out << std::scientific;
				out << std::setprecision(std::numeric_limits<T>::max_digits10);

				out << (MM_BANNER " " MM_MAT " " MM_FMT " " MM_TYPE_REAL " " MM_TYPE_GENERAL) << std::endl;
				out << rowN << " " << colN << " " << nnz << std::endl;

				for(size_t i=0; i<nnz; i++){
					out << row_index[i]+1 << " " << col_index[i]+1 << " " << val[i] << std::endl;
				}
				logger.util_out();
			}
		template void COO<double>::print_all(std::string filename);
		template void COO<float>::print_all(std::string filename);

		template<typename T>
			T COO<T>::at(size_t i, size_t j){
				Logger& logger = Logger::get_instance();
				logger.util_in(monolish_func);

				if(i >= rowN || j >= colN){
					throw std::out_of_range("error");
				}

				// since last inserted element is effective elements,
                                // checking from last element is necessary
                                for(size_t k = nnz; k > 0; --k){
					if( row_index[k-1] == (int)i && col_index[k-1] == (int)j){
						return val[k-1];
					}
				}
				logger.util_out();
				return 0.0;
			}
		template double COO<double>::at(size_t i, size_t j);
		template float COO<float>::at(size_t i, size_t j);

		template<typename T>
			void COO<T>::set_ptr(size_t rN, size_t cN, std::vector<int> &r, std::vector<int> &c, std::vector<T> &v){
				Logger& logger = Logger::get_instance();
				logger.util_in(monolish_func);
				col_index = c;
				row_index = r;
				val = v;

				rowN = rN;
				colN = cN;
				nnz = r.size();
				logger.util_out();
			}
			template void COO<double>::set_ptr(size_t rN, size_t cN, std::vector<int> &r, std::vector<int> &c, std::vector<double> &v);
			template void COO<float>::set_ptr(size_t rN, size_t cN, std::vector<int> &r, std::vector<int> &c, std::vector<float> &v);

        template<typename T>
        void COO<T>::insert(size_t m, size_t n, T value) {
            int rownum = m;
            if (rownum < 0 || rownum >= get_row()) { throw std::out_of_range("row index out of range"); }
            int colnum = n;
            if (colnum < 0 || colnum >= get_col()) { throw std::out_of_range("column index out of range"); }
            row_index.push_back(rownum);
            col_index.push_back(colnum);
            val.push_back(value);
            ++nnz;
        }
        template void COO<double>::insert(size_t m, size_t n, double value);
        template void COO<float>::insert(size_t m, size_t n, float value);

        template<typename T>
        void COO<T>::_q_sort(int lo, int hi) {
            // Very primitive quick sort
            if(lo >= hi) {
                return;
            }

            int l = lo;
            int h = hi;
            int p = hi;
            int p1 = row_index[p];
            int p2 = col_index[p];
            double p3 = val[p];

            do {
                while ((l < h) &&
                       ((row_index[l] != row_index[p])
                         ? (row_index[l] - row_index[p])
                         : (col_index[l] - col_index[p])) <= 0) {
                    l = l+1;
                }
                while ((h > l) &&
                       ((row_index[h] != row_index[p])
                        ? (row_index[h] - row_index[p])
                        : (col_index[h] - col_index[p])) >= 0) {
                    h = h-1;
                }
                if (l < h) {
                    int t = row_index[l];
                    row_index[l] = row_index[h];
                    row_index[h] = t;

                    t = col_index[l];
                    col_index[l] = col_index[h];
                    col_index[h] = t;

                    double td = val[l];
                    val[l] = val[h];
                    val[h] = td;
                }
            } while (l < h);

            row_index[p] = row_index[l];
            row_index[l] = p1;

            col_index[p] = col_index[l];
            col_index[l] = p2;

            val[p] = val[l];
            val[l] = p3;

            /* Sort smaller array first for less stack usage */
            if (l-lo < hi-l) {
                _q_sort(lo, l-1);
                _q_sort(l+1, hi);
            } else {
                _q_sort(l+1, hi);
                _q_sort(lo, l-1);
            }
        }
        template void COO<double>::_q_sort(int lo, int hi);
        template void COO<float>::_q_sort(int lo, int hi);

        template<typename T>
        void COO<T>::sort(bool merge) {
            //  Sort by first Col and then Row
            //  TODO: This hand-written quick sort function should be retired
            //        after zip_iterator() (available in range-v3 library) is available in the standard (hopefully C++23)
            _q_sort(0, nnz-1);

            /*  Remove duplicates */
            if (merge) {
                int k = 0;
                for( int i = 1; i < nnz; i++) {
                    if ((row_index[k] != row_index[i]) || (col_index[k] != col_index[i])) {
                        k++;
                        row_index[k] = row_index[i];
                        col_index[k] = col_index[i];
                    }
                    val[k] = val[i];
                }
                nnz = k+1;
            }
        }
        template void COO<double>::sort(bool merge = true);
        template void COO<float>::sort(bool merge = true);
	}
}
