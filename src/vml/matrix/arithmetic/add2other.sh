for func in sub mul div
do
  sed "s/add/${func}/g" matadd.hpp > mat${func}.hpp
done
