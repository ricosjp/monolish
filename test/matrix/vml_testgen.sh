  echo ""
for FUNC in msqrt msin msinh masin masinh mtan matan matanh mceil mfloor msign
do
  echo "// $FUNC Dense //"
  echo "if (test_send_$FUNC< monolish::matrix::Dense<double>, double>(M, N, 1.0e-8) == false) {return 1;}"
  echo "if (test_send_$FUNC<monolish::matrix::Dense<float>, float>(M, N, 1.0e-4) == false) {return 1;}"
  echo "if (test_$FUNC<monolish::matrix::Dense<double>, double>(M, N, 1.0e-8) == false) {return 1;}"
  echo "if (test_$FUNC<monolish::matrix::Dense<float>, float>(M, N, 1.0e-4) == false) {return 1;}"
  echo ""
done
