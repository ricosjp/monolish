#!/bin/bash

if [ $# != 2 ]; then
  echo "error!, \$1: func, \$2: arch."
  exit 1
fi

FORMAT=(Dense CRS)

case $1 in
  "sm_add" ) 
    for format in ${FORMAT[@]}; do
      echo start $1 $format $2
      $PROFILER ./$1_$2.out $format $format | tee $1\_$format\_$format\_$2.tsv 
      if [ ${PIPESTATUS[0]} -ne 0 ]; then
        exit 1
      fi
    done
    ;;
  "sm_sub" ) 
    for format in ${FORMAT[@]}; do
      echo start $1 $format $2
      $PROFILER ./$1_$2.out $format $format | tee $1\_$format\_$format\_$2.tsv 
      if [ ${PIPESTATUS[0]} -ne 0 ]; then
        exit 1
      fi
    done
    ;;
  "sm_mul" ) 
    for format in ${FORMAT[@]}; do
      echo start $1 $format $2
      $PROFILER ./$1_$2.out $format $format | tee $1\_$format\_$format\_$2.tsv 
      if [ ${PIPESTATUS[0]} -ne 0 ]; then
        exit 1
      fi
    done
    ;;
  "sm_div" ) 
    for format in ${FORMAT[@]}; do
      echo start $1 $format $2
      $PROFILER ./$1_$2.out $format $format | tee $1\_$format\_$format\_$2.tsv 
      if [ ${PIPESTATUS[0]} -ne 0 ]; then
        exit 1
      fi
    done
    ;;
  "mm_add" ) 
    for format in ${FORMAT[@]}; do
      echo start $1 $format $2
      $PROFILER ./$1_$2.out $format $format $format | tee $1\_$format\_$format\_$format\_$2.tsv 
      if [ ${PIPESTATUS[0]} -ne 0 ]; then
        exit 1
      fi
    done
    ;;
  "mm_sub" ) 
    for format in ${FORMAT[@]}; do
      echo start $1 $format $2
      $PROFILER ./$1_$2.out $format $format $format | tee $1\_$format\_$format\_$format\_$2.tsv 
      if [ ${PIPESTATUS[0]} -ne 0 ]; then
        exit 1
      fi
    done
    ;;
  "mm_mul" ) 
    for format in ${FORMAT[@]}; do
      echo start $1 $format $2
      $PROFILER ./$1_$2.out $format $format $format | tee $1\_$format\_$format\_$format\_$2.tsv 
      if [ ${PIPESTATUS[0]} -ne 0 ]; then
        exit 1
      fi
    done
    ;;
  "mm_div" ) 
    for format in ${FORMAT[@]}; do
      echo start $1 $format $2
      $PROFILER ./$1_$2.out $format $format $format | tee $1\_$format\_$format\_$format\_$2.tsv 
      if [ ${PIPESTATUS[0]} -ne 0 ]; then
        exit 1
      fi
    done
    ;;
  "m_sqrt" ) 
    for format in ${FORMAT[@]}; do
      echo start $1 $format $2
      $PROFILER ./$1_$2.out $format | tee $1\_$format\_$2.tsv 
      if [ ${PIPESTATUS[0]} -ne 0 ]; then
        exit 1
      fi
    done
    ;;
  "mm_pow" ) 
    for format in ${FORMAT[@]}; do
      echo start $1 $format $2
      $PROFILER ./$1_$2.out $format $format $format | tee $1\_$format\_$format\_$format\_$2.tsv 
      if [ ${PIPESTATUS[0]} -ne 0 ]; then
        exit 1
      fi
    done
    ;;
  "sm_pow" ) 
    for format in ${FORMAT[@]}; do
      echo start $1 $format $2
      $PROFILER ./$1_$2.out $format $format | tee $1\_$format\_$format\_$2.tsv 
      if [ ${PIPESTATUS[0]} -ne 0 ]; then
        exit 1
      fi
    done
    ;;
  "m_sin" ) 
    for format in ${FORMAT[@]}; do
      echo start $1 $format $2
      $PROFILER ./$1_$2.out $format | tee $1\_$format\_$2.tsv 
      if [ ${PIPESTATUS[0]} -ne 0 ]; then
        exit 1
      fi
    done
    ;;
  "m_sinh" ) 
    for format in ${FORMAT[@]}; do
      echo start $1 $format $2
      $PROFILER ./$1_$2.out $format | tee $1\_$format\_$2.tsv 
      if [ ${PIPESTATUS[0]} -ne 0 ]; then
        exit 1
      fi
    done
    ;;
  "m_asin" ) 
    for format in ${FORMAT[@]}; do
      echo start $1 $format $2
      $PROFILER ./$1_$2.out $format | tee $1\_$format\_$2.tsv 
      if [ ${PIPESTATUS[0]} -ne 0 ]; then
        exit 1
      fi
    done
    ;;
  "m_asinh" ) 
    for format in ${FORMAT[@]}; do
      echo start $1 $format $2
      $PROFILER ./$1_$2.out $format | tee $1\_$format\_$2.tsv 
      if [ ${PIPESTATUS[0]} -ne 0 ]; then
        exit 1
      fi
    done
    ;;
  "m_tan" ) 
    for format in ${FORMAT[@]}; do
      echo start $1 $format $2
      $PROFILER ./$1_$2.out $format | tee $1\_$format\_$2.tsv 
      if [ ${PIPESTATUS[0]} -ne 0 ]; then
        exit 1
      fi
    done
    ;;
  "m_tanh" ) 
    for format in ${FORMAT[@]}; do
      echo start $1 $format $2
      $PROFILER ./$1_$2.out $format | tee $1\_$format\_$2.tsv 
      if [ ${PIPESTATUS[0]} -ne 0 ]; then
        exit 1
      fi
    done
    ;;
  "m_atan" ) 
    for format in ${FORMAT[@]}; do
      echo start $1 $format $2
      $PROFILER ./$1_$2.out $format | tee $1\_$format\_$2.tsv 
      if [ ${PIPESTATUS[0]} -ne 0 ]; then
        exit 1
      fi
    done
    ;;
  "m_atanh" ) 
    for format in ${FORMAT[@]}; do
      echo start $1 $format $2
      $PROFILER ./$1_$2.out $format | tee $1\_$format\_$2.tsv 
      if [ ${PIPESTATUS[0]} -ne 0 ]; then
        exit 1
      fi
    done
    ;;
  "m_ceil" ) 
    for format in ${FORMAT[@]}; do
      echo start $1 $format $2
      $PROFILER ./$1_$2.out $format | tee $1\_$format\_$2.tsv 
      if [ ${PIPESTATUS[0]} -ne 0 ]; then
        exit 1
      fi
    done
    ;;
  "m_floor" ) 
    for format in ${FORMAT[@]}; do
      echo start $1 $format $2
      $PROFILER ./$1_$2.out $format | tee $1\_$format\_$2.tsv 
      if [ ${PIPESTATUS[0]} -ne 0 ]; then
        exit 1
      fi
    done
    ;;
  "m_sign" ) 
    for format in ${FORMAT[@]}; do
      echo start $1 $format $2
      $PROFILER ./$1_$2.out $format | tee $1\_$format\_$2.tsv 
      if [ ${PIPESTATUS[0]} -ne 0 ]; then
        exit 1
      fi
    done
    ;;
  "m_reciprocal" ) 
    for format in ${FORMAT[@]}; do
      echo start $1 $format $2
      $PROFILER ./$1_$2.out $format | tee $1\_$format\_$2.tsv 
      if [ ${PIPESTATUS[0]} -ne 0 ]; then
        exit 1
      fi
    done
    ;;
esac
