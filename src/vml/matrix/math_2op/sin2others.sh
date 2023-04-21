for func in sinh asin asinh tan tanh atan atanh sqrt ceil floor sign reciprocal exp
do
  sed "s/sin/${func}/g" matsin.hpp > mat${func}.hpp
done
