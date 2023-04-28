for func in sub mul div
do
  sed "s/add/${func}/g" tensadd.hpp > tens${func}.hpp
done
