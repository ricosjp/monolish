  echo ""
for FUNC in vvpow svpow vsqrt vsin vsinh vasin vasinh vtan vatan vatanh vceil vfloor vsign
do
  echo "// $FUNC //"
  echo "if (test_$FUNC<double>(size, 1.0e-8) == false) {return 1;}"
  echo "if (test_$FUNC<float>(size, 1.0e-4) == false) {return 1;}"
  echo "if (test_send_$FUNC<double>(size, 1.0e-8) == false) {return 1;}"
  echo "if (test_send_$FUNC<float>(size, 1.0e-4) == false) {return 1;}"
  echo ""
done
