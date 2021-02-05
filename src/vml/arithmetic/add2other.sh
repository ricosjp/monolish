for func in sub mul div
do
  sed "s/add/${func}/g" vvadd.cpp > vv${func}.cpp
  sed "s/add/${func}/g" svadd.cpp > sv${func}.cpp
  sed "s/add/${func}/g" mmadd.cpp > mm${func}.cpp
  sed "s/add/${func}/g" smadd.cpp > sm${func}.cpp
done
