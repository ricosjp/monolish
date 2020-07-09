#include "../../include/common/monolish_matrix.hpp" 
#include "../../include/common/monolish_logger.hpp" 
#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

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
				size_t rowN, colN, NNZ;

				std::istringstream data(buf);
				data >> rowN >> colN >> NNZ;

				//symmetric check!
				if (colN != rowN) {
					std::cerr << "Matrix.input: Matrix is not square" << std::endl;
					exit(-1);
				}
				if (colN <= 0 || NNZ < 0) {
					std::cerr << "Matrix.input: Matrix size should be positive" << std::endl;
					exit(-1);
				}

				row = rowN;
				col = row;
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
			void COO<T>::output_mm(const char* filename){
				Logger& logger = Logger::get_instance();
				logger.util_in(monolish_func);
				std::ofstream out(filename);
				out << std::scientific;
				out << std::setprecision(std::numeric_limits<T>::max_digits10);

				out << (MM_BANNER " " MM_MAT " " MM_FMT " " MM_TYPE_REAL " " MM_TYPE_GENERAL) << std::endl;
				out << row << " " << row << " " << nnz << std::endl;

				for(size_t i=0; i<nnz; i++){
					out << row_index[i]+1 << " " << col_index[i]+1 << " " << val[i] << std::endl;
				}
				logger.util_out();
			}
		template void COO<double>::output_mm(const char* filename);
		template void COO<float>::output_mm(const char* filename);

		template<typename T>
			void COO<T>::output(){
				Logger& logger = Logger::get_instance();
				logger.util_in(monolish_func);
				for(size_t i=0; i<nnz; i++){
					std::cout << row_index[i]+1 << " " << col_index[i]+1 << " " << val[i] << std::endl;
				}
				logger.util_out();
			}
		template void COO<double>::output();
		template void COO<float>::output();

		template<typename T>
			T COO<T>::at(size_t i, size_t j){
				Logger& logger = Logger::get_instance();
				logger.util_in(monolish_func);

				if(i >= row || j >= col){
					throw std::out_of_range("error");
				}

				for(size_t i=0; i<nnz; i++){
					if( row_index[i] == (int)i && col_index[i] == (int)j){
						return val[i];
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

				row = rN;
				col = cN;
				nnz = r.size();
				logger.util_out();
			}
			template void COO<double>::set_ptr(size_t rN, size_t cN, std::vector<int> &r, std::vector<int> &c, std::vector<double> &v);
			template void COO<float>::set_ptr(size_t rN, size_t cN, std::vector<int> &r, std::vector<int> &c, std::vector<float> &v);

        template<typename T>
        void COO<T>::insert(size_t m, size_t n, T value, size_t offset) {
            int rownum = m - offset;
            if (rownum < 0 || rownum >= get_row()) { throw std::out_of_range("row index out of range"); }
            int colnum = n - offset;
            if (colnum < 0 || colnum >= get_col()) { throw std::out_of_range("column index out of range"); }
            row_index.push_back(rownum);
            col_index.push_back(colnum);
            val.push_back(value);
        }
        template void COO<double>::insert(size_t m, size_t n, double value, size_t offset = 0);
        template void COO<float>::insert(size_t m, size_t n, float value, size_t offset = 0);

        template<typename T>
        void COO<T>::sort(bool merge) {

        }
        template void COO<double>::sort(bool merge = true);
        template void COO<float>::sort(bool merge = true);
	}
}
