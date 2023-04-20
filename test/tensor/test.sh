#!/bin/bash

if [ $# != 2 ]; then
  echo "error!, \$1: func, \$2: arch."
  exit 1
fi

FORMAT=(Dense)
PREC=(double float)
TEST_A=$(($RANDOM%50+50)) #50~100
TEST_B=$(($RANDOM%50+50)) #50~100
TEST_C=$(($RANDOM%50+50)) #50~100

case $1 in
  "tensor_common" )
    $PROFILER ./$1_$2.out $prec $format $M $N $L 1 || exit 1
    ;;
esac

