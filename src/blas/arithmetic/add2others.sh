cat add.cpp | sed -e 's/vadd/vsub/g' | sed -e 's;operator+;operator-;g' > sub.cpp
cat add.cpp | sed -e 's/vadd/vmul/g' | sed -e 's;operator+;operator*;g' > mul.cpp
cat add.cpp | sed -e 's/vadd/vdiv/g' | sed -e 's;operator+;operator/;g' > div.cpp
