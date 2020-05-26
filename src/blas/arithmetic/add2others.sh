cat add.cpp | sed -e 's; +; -;g' | sed -e 's;operator+;operator-;g'  > sub.cpp
cat add.cpp | sed -e 's; +; *;g' | sed -e 's;operator+;operator*;g'  > mul.cpp
cat add.cpp | sed -e 's; +; /;g' | sed -e 's;operator+;operator/;g'  > div.cpp
