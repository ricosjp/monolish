#!/bin/bash

## allreduce
for prec in double float int size_t; do
  echo "
  /**
  * @brief MPI_Allreduce (MPI_SUM) for scalar. Combines values from all processes and distributes the result back to all processes.
  * @param val scalar value
  */ "
  echo "$prec Allreduce($prec val);"
done

for prec in double float int size_t; do
  echo "
  /**
  * @brief MPI_Allreduce (MPI_SUM) for scalar. Combines values from all processes and distributes the result back to all processes.
  * @param val scalar value
  */ "
  echo "$prec Allreduce_sum($prec val);"
done

for prec in double float int size_t; do
  echo "
  /**
  * @brief MPI_Allreduce (MPI_PROD) for scalar. Combines values from all processes and distributes the result back to all processes.
  * @param val scalar value
  */ "
  echo "$prec Allreduce_prod($prec val);"
done

for prec in double float int size_t; do
  echo "
  /**
  * @brief MPI_Allreduce (MPI_MAX) for scalar. Combines values from all processes and distributes the result back to all processes.
  * @param val scalar value
  */ "
  echo "$prec Allreduce_max($prec val);"
done

for prec in double float int size_t; do
  echo "
  /**
  * @brief MPI_Allreduce (MPI_MIN) for scalar. Combines values from all processes and distributes the result back to all processes.
  * @param val scalar value
  */ "
  echo "$prec Allreduce_min($prec val);"
done
