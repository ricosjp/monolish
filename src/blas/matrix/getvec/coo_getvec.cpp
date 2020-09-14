#include "../../../../include/monolish_blas.hpp"
#include "../../../monolish_internal.hpp"

namespace monolish{
	namespace matrix{

		//diag
		template<typename T>
		std::vector<T> COO<T>::diag() const {
			Logger& logger = Logger::get_instance();
			logger.func_in(monolish_func);

                        std::size_t s = get_row() > get_col() ? get_col() : get_row();
                        std::vector<T> res(s, 0);
                        for (std::size_t nz = 0; nz < get_nnz(); ++nz) {
                            if (get_row_ptr()[nz] == get_col_ind()[nz]) {
			      res[get_row_ptr()[nz]] = get_val_ptr()[nz];
                            }
                        }
			logger.func_out();
                        return res;
		}
	  template std::vector<double> monolish::matrix::COO<double>::diag() const;
	  template std::vector<float>  monolish::matrix::COO<float>::diag() const;

		template<typename T>
		void COO<T>::diag(vector<T>& vec) const {
			Logger& logger = Logger::get_instance();
			logger.func_in(monolish_func);

                        std::size_t s = get_row() > get_col() ? get_col() : get_row();
                        vec.resize(s);
                        for (std::size_t nz = 0; nz < get_nnz(); ++nz) {
                            if (get_row_ptr()[nz] == get_col_ind()[nz]) {
                                vec[get_row_ptr()[nz]] = get_val_ptr()[nz];
                            }
                        }
			logger.func_out();
		}
		template void monolish::matrix::COO<double>::diag(vector<double>& vec) const;
		template void monolish::matrix::COO<float>::diag(vector<float>& vec) const;

		//row
		template<typename T>
		std::vector<T> COO<T>::row(const size_t r) const {
			Logger& logger = Logger::get_instance();
			logger.func_in(monolish_func);

                        std::vector<T> res(get_col(), 0);
                        for (std::size_t nz = 0; nz < get_nnz(); ++nz) {
			  if (get_row_ptr()[nz] == static_cast<int>(r)) {
                                res[get_col_ind()[nz]] = get_val_ptr()[nz];
                            }
                        }
			logger.func_out();
                        return res;
		}
	  template std::vector<double> monolish::matrix::COO<double>::row(const size_t r) const;
	  template std::vector<float> monolish::matrix::COO<float>::row(const size_t r) const;

		template<typename T>
		void COO<T>::row(const size_t r, vector<T>& vec) const {
			Logger& logger = Logger::get_instance();
			logger.func_in(monolish_func);

                        vec.resize(get_col());
                        for (std::size_t nz = 0; nz < get_nnz(); ++nz) {
                            if (get_row_ptr()[nz] == static_cast<int>(r)) {
                                vec[get_col_ind()[nz]] = get_val_ptr()[nz];
                            }
                        }
			logger.func_out();
		}
		template void monolish::matrix::COO<double>::row(const size_t r, vector<double>& vec) const;
		template void monolish::matrix::COO<float>::row(const size_t r, vector<float>& vec) const;

		//col
		template<typename T>
		std::vector<T> COO<T>::col(const size_t c) const {
			Logger& logger = Logger::get_instance();
			logger.func_in(monolish_func);

                        std::vector<T> res(get_row(), 0);
                        for (std::size_t nz = 0; nz < get_nnz(); ++nz) {
                            if (get_col_ind()[nz] == static_cast<int>(c)) {
                                res[get_row_ptr()[nz]] = get_val_ptr()[nz];
                            }
                        }
			logger.func_out();
                        return res;
		}
	  template std::vector<double> monolish::matrix::COO<double>::col(const size_t c) const;
	  template std::vector<float> monolish::matrix::COO<float>::col(const size_t c) const;

		template<typename T>
		void COO<T>::col(const size_t c, vector<T>& vec) const{
			Logger& logger = Logger::get_instance();
			logger.func_in(monolish_func);

                        vec.resize(get_row());
                        for (std::size_t nz = 0; nz < get_nnz(); ++nz) {
                            if (get_col_ind()[nz] == static_cast<int>(c)) {
                                vec[get_row_ptr()[nz]] = get_val_ptr()[nz];
                            }
                        }
			logger.func_out();
		}
		template void monolish::matrix::COO<double>::col(const size_t c, vector<double>& vec) const;
		template void monolish::matrix::COO<float>::col(const size_t c, vector<float>& vec) const;
	}
}
