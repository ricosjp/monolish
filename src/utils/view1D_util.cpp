#include "../../include/monolish_blas.hpp"
#include "../../include/monolish_vml.hpp"
#include "../internal/monolish_internal.hpp"
namespace monolish {
  template<> auto* view1D<monolish::vector<double>>::data(){
    return target.data()+first;
  }

  template<> auto* view1D<monolish::matrix::Dense<double>>::data(){
    return target.val.data()+first;
  }

  template<> void view1D<monolish::vector<double>>::print_all() const{
    for(size_t i=first; i<last; i++){
      std::cout << target[i] << std::endl;
    }
  }

  template<> void view1D<monolish::matrix::Dense<double>>::print_all() const{
    for(size_t i=first; i<last; i++){
      std::cout << target.val[i] << std::endl;
    }
  }

  template<> auto& view1D<monolish::vector<double>>::operator[](const size_t i){
    if (get_device_mem_stat()) {
      throw std::runtime_error("Error, GPU vector cant use operator[]");
    }
    return target[first+i];
  }

  template<> auto& view1D<monolish::matrix::Dense<double>>::operator[](const size_t i){
    if (get_device_mem_stat()) {
      throw std::runtime_error("Error, GPU vector cant use operator[]");
    }
    return target.val[first+i];
  }

  template<> void view1D<monolish::vector<double>>::resize(size_t N){
    assert(first+N <= target.size());
    last = first + N;
  }

  template<> void view1D<monolish::matrix::Dense<double>>::resize(size_t N){
    assert(first+N <= target.get_nnz());
    last = first + N;
  }

  template<> void view1D<monolish::vector<double>>::set_first(size_t i){
    first=i;
  }
  template<> void view1D<monolish::matrix::Dense<double>>::set_first(size_t i){
    first=i;
  }

  template<> void view1D<monolish::vector<double>>::set_last(size_t i){
    assert(first+i <= target.size());
    last=i;
  }
  template<> void view1D<monolish::matrix::Dense<double>>::set_last(size_t i){
    assert(first+i <= target.get_nnz());
    last=i;
  }

}
