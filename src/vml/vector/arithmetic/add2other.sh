for func in sub mul div
do
  sed "s/add/${func}/g" vecadd.hpp > vec${func}.hpp
done
