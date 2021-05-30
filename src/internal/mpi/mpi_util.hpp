#if defined MONOLISH_USE_MPI
namespace monolish::internal::mpi {
  template<typename T>
  auto get_type(T val){
  if (typeid(double) == typeid(val)) {
    return MPI_DOUBLE;
  }
  if (typeid(float) == typeid(val)) {
    return MPI_FLOAT;
  }
  if (typeid(int) == typeid(val)) {
    return MPI_INT;
  }
  if (typeid(size_t) == typeid(val)) {
    return MPI_SIZE_T
  }
  assert();
  }
}
#endif
