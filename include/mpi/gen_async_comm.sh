#/bin/bash

## Isend
for prec in double float int size_t; do
  echo "
  /**
  * @brief MPI_Isend for scalar. Performs a nonblocking send. Requests are stored internally. All requests are synchronized by Waitall().
  * @param val scalar value
  * @param dst rank of dstination
  * @param tag message tag
  * @Warning
  * This function send data asynchronously. There is not MPI wait() functions in monolish.
  * Asynchronous send data is synchronized in the recv() function.
  * This function does not support GPU.
  * The user needs to communicate with the GPU before and after the call to this function if necessary.
  */"
  echo "MPI_Request send($prec val, int dst, int tag, bool only_cpu = false) const;"
done

for prec in double float int size_t; do
  echo "
  /**
  * @brief MPI_Isend for std::vector. Performs a nonblocking send. Requests are stored internally. All requests are synchronized by Waitall().
  * @param vec std::vector (size N)
  * @param dst rank of dstination
  * @param tag message tag
  */"
  echo "MPI_Request send(std::vector<$prec> &vec, int dst, int tag, bool only_cpu = false) const;"
done

for prec in double float; do
  echo "
  /**
  * @brief send for monolish::vector. Performs a non-blocking send.
  * @param vec std::vector (size N)
  * @param dst rank of dstination
  * @param tag message tag
  * @warning
  * When "only_cpu flag is enabled" and "send data is on the GPU", data is received from the GPU, 
  * MPI communication is performed, and finally data is sent to the GPU.
  * If there is no GPU, or if there is no data on the GPU, no error will occur even if this flag is set to true.
  */ "
  echo "MPI_Request send(monolish::vector<$prec> &vec, int dst, int tag, bool only_cpu = false) const;"
done


## Irecv
for prec in double float int size_t; do
  echo "
  /**
  * @brief MPI_Irecv for scalar. Performs a nonblocking recv.
  * @param val scalar value
  * @param src rank of source
  * @param tag message tag
  */"
  echo "void Irecv($prec val, int src, int tag, bool only_cpu = false) const;"
done

for prec in double float int size_t; do
  echo "
  /**
  * @brief MPI_Irecv for std::vector. Performs a nonblocking recv.
  * @param vec std::vector (size N)
  * @param src rank of source
  * @param tag message tag
  */"
  echo "void Irecv(std::vector<$prec> &vec, int src, int tag, bool only_cpu = false) const;"
done

for prec in double float; do
  echo "
  /**
  * @brief MPI_Irecv for monolish::vector. Performs a nonblocking recv.
  * @param vec monolish::vector (size N)
  * @param src rank of source
  * @param tag message tag
  * @warning
  * When "only_cpu flag is enabled" and "recv data is on the GPU", data is received from the GPU, 
  * MPI communication is performed, and finally data is sent to the GPU.
  * If there is no GPU, or if there is no data on the GPU, no error will occur even if this flag is set to true.
  */ "
  echo "void Irecv(monolish::vector<$prec> &vec, int src, int tag, bool only_cpu = false) const;"
done

echo "
/**
* @brief Waits for all communications to complete.
*/ "
echo "void Waitall() const;"
