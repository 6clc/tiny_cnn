# g++ -Wall -o test $@  -I/opt/intel/mkl/include -I/usr/include/eigen3/  
# ./test

# g++ -Wall -c  PooLayerQL.cc -I/opt/intel/mkl/include -I/usr/include/eigen3/  
# g++ -Wall -c  lc_tests/Activation_Test.cc -I/opt/intel/mkl/include -I/usr/include/eigen3/  

g++ -Wall -o test $@  -I/usr/include/mkl -I/usr/include/eigen3/  
./test