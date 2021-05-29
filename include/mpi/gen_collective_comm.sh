#/bin/bash

## Bcast
for prec in double float int size_t; do
  echo "
  /**
   * @brief MPI_Bcast, Broadcasts a message from the process with rank "root" to all other processes
   * @param val scalar value
   * @param val root rank number
   */
  void Bcast($prec &val, int root) const;"
done

for prec in double float; do
  echo "
  /**
   * @brief MPI_Bcast, Broadcasts a message from the process with rank "root" to all other processes
   * @param monolish vector (size N)
   * @param val root rank number
   */
  void Bcast(monolish::vector<$prec> &val, int root) const;"
done
