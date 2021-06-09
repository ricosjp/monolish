#include "monolish_mpi.hpp"

template <typename T> T test_sum(std::vector<T> &vec) {

  monolish::mpi::comm &comm = monolish::mpi::comm::get_instance();
  double sum = 0;

  for (size_t i = 0; i < vec.size(); i++) {
    sum += vec[i];
  }

  return comm.Allreduce(sum);
}

int main(int argc, char **argv) {

  if (argc != 2) {
    std::cout << "error!, $1:vector size" << std::endl;
    return 1;
  }

  size_t size = atoi(argv[1]);
  std::cout << "size: " << size << std::endl;

  // monolish::util::set_log_level(3);
  // monolish::util::set_log_filename("./monolish_test_log.txt");

  monolish::mpi::comm &comm = monolish::mpi::comm::get_instance();
  comm.Init(argc, argv);

  int rank = comm.get_rank();
  int procs = comm.get_size();

  std::cout << "I am" << rank << "/" << procs << std::endl;

  std::vector<double> dvec(size, 1);
  std::vector<float> fvec(size, 1);
  std::vector<int> ivec(size, 1);
  std::vector<size_t> svec(size, 1);

  comm.Barrier();

  if (test_sum(dvec) != size * procs) {
    std::cout << "error in double" << std::endl;
  }
  if (test_sum(fvec) != size * procs) {
    std::cout << "error in float" << std::endl;
  }
  if (test_sum(ivec) != size * procs) {
    std::cout << "error in int" << std::endl;
  }
  if (test_sum(svec) != size * procs) {
    std::cout << "error in size_t" << std::endl;
  }

  comm.Finalize();

  return 0;
}
