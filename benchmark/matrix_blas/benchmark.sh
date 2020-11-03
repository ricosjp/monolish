#!/bin/bash

if [ $# != 2 ]; then
  echo "error!, \$1: func, \$2: arch."
  exit 1
fi

FORMAT=(Dense CRS)
PREC=(double float)

case $1 in
  "matadd" ) 
    for format in ${FORMAT[@]}; do
      echo start $1 $format $2
      $PROFILER ./$1_$2.out all $format $format $format | tee $1\_$format\_$format\_$format\_$2.tsv
    done
    ;;
  "mscal" ) 
    for format in ${FORMAT[@]}; do
      echo start $1 $format $2
      $PROFILER ./$1_$2.out all $format | tee $1\_$format\_$2.tsv
    done
    ;;
  "matvec" ) 
    for format in ${FORMAT[@]}; do
      echo start $1 $format $2
      $PROFILER ./$1_$2.out all $format | tee $1\_$format\_$2.tsv
    done
    ;;
  "matmul" ) 
    for format in ${FORMAT[@]}; do
      echo start $1 $format $2
      $PROFILER ./$1_$2.out all $format Dense Dense | tee $1\_$format\_Dense\_Dense\_$2.tsv
    done
    ;;
esac
