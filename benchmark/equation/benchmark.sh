#!/bin/bash

if [ $# != 2 ]; then
  echo "error!, \$1: func, \$2: arch."
  exit 1
fi

FORMAT=(Dense)

case $1 in
  "LU" ) 
    for format in Dense; do
      echo start $1 $format $2
      $PROFILER ./$1_$2.out $format  | tee $1\_$format\_$2.tsv 
      if [ ${PIPESTATUS[0]} -ne 0 ]; then
        exit 1
      fi
    done
    ;;
  "CG" ) 
    for format in CRS; do
      echo start $1 $format $2
      $PROFILER ./$1_$2.out $format | tee $1\_$format\_$2.tsv 
      if [ ${PIPESTATUS[0]} -ne 0 ]; then
        exit 1
      fi
    done
    ;;
esac
