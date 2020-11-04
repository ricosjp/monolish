#!/bin/bash

if [ $# != 2 ]; then
  echo "error!, \$1: func, \$2: arch."
  exit 1
fi

FORMAT=(Dense CRS)

case $1 in
  "convert" ) 
    for format in ${FORMAT[@]}; do
      echo start $1 $format $2
      $PROFILER ./$1_$2.out $format | tee $1\_$format\_$2.tsv
    done
    ;;
  "mv_mul" ) 
    for format in ${FORMAT[@]}; do
      echo start $1 $format $2
      $PROFILER ./$1_$2.out $format | tee $1\_$format\_$2.tsv
    done
    ;;
  "transpose" ) 
    for format in ${FORMAT[@]}; do
      echo start $1 $format $2
      $PROFILER ./$1_$2.out $format | tee $1\_$format\_$2.tsv
    done
    ;;
  "m_tanh" ) 
    for format in ${FORMAT[@]}; do
      echo start $1 $format $2
      $PROFILER ./$1_$2.out $format | tee $1\_$format\_$2.tsv
    done
    ;;
esac
