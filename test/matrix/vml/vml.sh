cat ./m_tanh.hpp | sed -e 's/tanh/sqrt/g' > m_sqrt.hpp
cat ./m_tanh.hpp | sed -e 's/tanh/sin/g' > m_sin.hpp
cat ./m_tanh.hpp | sed -e 's/tanh/sinh/g' > m_sinh.hpp
cat ./m_tanh.hpp | sed -e 's/tanh/asin/g' > m_asin.hpp
cat ./m_tanh.hpp | sed -e 's/tanh/asinh/g' > m_asinh.hpp
cat ./m_tanh.hpp | sed -e 's/tanh/tan/g' > m_tan.hpp
cat ./m_tanh.hpp | sed -e 's/tanh/atan/g' > m_atan.hpp
cat ./m_tanh.hpp | sed -e 's/tanh/atanh/g' > m_atanh.hpp
cat ./m_tanh.hpp | sed -e 's/tanh/ceil/g' > m_ceil.hpp
cat ./m_tanh.hpp | sed -e 's/tanh/floor/g' > m_floor.hpp
#cat ./m_tanh.hpp | sed -e 's/tanh/sign/g' > m_sign.hpp
cat ./m_tanh.hpp | sed -e 's/tanh/exp/g' > m_exp.hpp
cat ./m_max.hpp | sed -e 's/max/min/g' > m_min.hpp
cat ./mm_max.hpp | sed -e 's/max/min/g' > mm_min.hpp
cat ./sm_max.hpp | sed -e 's/max/min/g' > sm_min.hpp
#cat ./mm_add.hpp | sed -e 's/add/sub/g' > mm_sub.hpp
#cat ./mm_add.hpp | sed -e 's/add/mul/g' > mm_mul.hpp
#cat ./mm_add.hpp | sed -e 's/add/div/g' > mm_div.hpp
#cat ./mm_add.hpp | sed -e 's/add/pow/g' > mm_pow.hpp
#cat ./sm_add.hpp | sed -e 's/add/sub/g' > sm_sub.hpp
#cat ./sm_add.hpp | sed -e 's/add/mul/g' > sm_mul.hpp
#cat ./sm_add.hpp | sed -e 's/add/div/g' > sm_div.hpp
#cat ./sm_add.hpp | sed -e 's/add/pow/g' > sm_pow.hpp
