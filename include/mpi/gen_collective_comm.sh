#/bin/bash

## allreduce
for prec in double float int size_t; do
  echo "
  /**
  * @brief MPI_Allreduce (MPI_SUM) for scalar. Combines values from all processes and distributes the result back to all processes.
  * @param val scalar value
  */ "
  echo "[[nodiscard]] $prec Allreduce($prec val) const;"
done

for prec in double float int size_t; do
  echo "
  /**
  * @brief MPI_Allreduce (MPI_SUM) for scalar. Combines values from all processes and distributes the result back to all processes.
  * @param val scalar value
  */ "
  echo "[[nodiscard]] $prec Allreduce_sum($prec val) const;"
done

for prec in double float int size_t; do
  echo "
  /**
  * @brief MPI_Allreduce (MPI_PROD) for scalar. Combines values from all processes and distributes the result back to all processes.
  * @param val scalar value
  */ "
  echo "[[nodiscard]] $prec Allreduce_prod($prec val) const;"
done

for prec in double float int size_t; do
  echo "
  /**
  * @brief MPI_Allreduce (MPI_MAX) for scalar. Combines values from all processes and distributes the result back to all processes.
  * @param val scalar value
  */ "
  echo "[[nodiscard]] $prec Allreduce_max($prec val) const;"
done

for prec in double float int size_t; do
  echo "
  /**
  * @brief MPI_Allreduce (MPI_MIN) for scalar. Combines values from all processes and distributes the result back to all processes.
  * @param val scalar value
  */ "
  echo "[[nodiscard]] $prec Allreduce_min($prec val) const;"
done

## Bcast
for prec in double float int size_t; do
  echo "
  /**
  * @brief MPI_Bcast, Broadcasts a message from the process with rank "root" to all other processes
  * @param val scalar value
  * @param root root rank number
  */
  void Bcast($prec &val, int root) const;"
done

for prec in double float; do
  echo "
  /**
  * @brief MPI_Bcast, Broadcasts a message from the process with rank "root" to all other processes
  * @param vec monolish vector (size N)
  * @param root root rank number
  */
  void Bcast(monolish::vector<$prec> &vec, int root) const;"
done

for prec in double float int size_t; do
  echo "
  /**
  * @brief MPI_Bcast, Broadcasts a message from the process with rank "root" to all other processes
  * @param vec std::vector (size N)
  * @param root root rank number
  */
  void Bcast(std::vector<$prec> &vec, int root) const;"
done

## Gather
for prec in double float; do
  echo "
  /**
  * @brief MPI_Gather, Gathers vector from all processes
  * The data is evenly divided and transmitted to each process.
  * @param sendvec send data, monolish vector (size N)
  * @param recvvec recv data, monolish vector (size N * # of procs)
  * @param val root rank number
  * @warning
  * MPI functions do not support GPUs.
  * The user needs to send and receive data to and from the GPU before and after the MPI function.
  */
  void Gather(monolish::vector<$prec> &sendvec, monolish::vector<$prec> &recvvec, int root) const;"
done

for prec in double float int size_t; do
  echo "
  /**
  * @brief MPI_Gather, Gathers vector from all processes
  * The data is evenly divided and transmitted to each process.
  * @param sendvec send data, std vector (size N)
  * @param recvvec recv data, std vector (size N * # of procs)
  * @param val root rank number
  */
  void Gather(std::vector<$prec> &sendvec, std::vector<$prec> &recvvec, int root) const;"
done

## Scatter
for prec in double float; do
  echo "
  /**
  * @brief MPI_Scatter, Sends data from one task to all tasks.
  * The data is evenly divided and transmitted to each process.
  * @param sendvec send data, monolish vector (size N)
  * @param recvvec recv data, monolish vector (size N / # of procs)
  * @param val root rank number
  * @warning
  * MPI functions do not support GPUs.
  * The user needs to send and receive data to and from the GPU before and after the MPI function.
  */
  void Scatter(monolish::vector<$prec> &sendvec, monolish::vector<$prec> &recvvec, int root) const;"
done

for prec in double float int size_t; do
  echo "
  /**
  * @brief MPI_Scatter, Sends data from one task to all tasks.
  * The data is evenly divided and transmitted to each process.
  * @param sendvec send data, std::vector (size N)
  * @param recvvec recv data, std::vector (size N / # of procs)
  * @param val root rank number
  */
  void Scatter(std::vector<$prec> &sendvec, std::vector<$prec> &recvvec, int root) const;"
done
