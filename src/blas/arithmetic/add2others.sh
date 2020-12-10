cat add.cpp | sed -e 's/add/sub/g' | sed -e 's;operator+;operator-;g' > sub.cpp
cat add.cpp | sed -e 's/add/mul/g' | sed -e 's;operator+;operator*;g' > mul.cpp
cat add.cpp | sed -e 's/add/div/g' | sed -e 's;operator+;operator/;g' > div.cpp
