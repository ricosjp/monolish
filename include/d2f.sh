sed 's/double precision/single precision/g' monolish_vml_double.hpp | \
    sed 's/double/float/g' > monolish_vml_float.hpp
