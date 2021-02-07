cat ./vtanh.hpp | sed -e 's/tanh/sqrt/g' > vsqrt.hpp
cat ./vtanh.hpp | sed -e 's/tanh/sin/g' > vsin.hpp
cat ./vtanh.hpp | sed -e 's/tanh/sinh/g' > vsinh.hpp
cat ./vtanh.hpp | sed -e 's/tanh/asin/g' > vasin.hpp
cat ./vtanh.hpp | sed -e 's/tanh/asinh/g' > vasinh.hpp
cat ./vtanh.hpp | sed -e 's/tanh/tan/g' > vtan.hpp
cat ./vtanh.hpp | sed -e 's/tanh/atan/g' > vatan.hpp
cat ./vtanh.hpp | sed -e 's/tanh/atanh/g' > vatanh.hpp
cat ./vtanh.hpp | sed -e 's/tanh/ceil/g' > vceil.hpp
cat ./vtanh.hpp | sed -e 's/tanh/floor/g' > vfloor.hpp
cat ./vtanh.hpp | sed -e 's/tanh/sign/g' > vsign.hpp

echo ""
