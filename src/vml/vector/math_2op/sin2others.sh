for func in sinh asin asinh tan tanh atan atanh sqrt ceil floor sign reciprocal exp
do
  sed "s/sin/${func}/g" vecsin.hpp > vec${func}.hpp
done
