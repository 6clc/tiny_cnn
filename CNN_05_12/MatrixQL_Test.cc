# include "MatrixQL.h"

#include <stdio.h>
#include <iostream>
using namespace std;

int main(int argc, char* argv[]){
  tinyDNN::MatrixQL<int> m(2, 2);
  auto cur_data = m.getMatrixQL();
  cout << cur_data << endl;
  
  auto cur_data_set = m.setMatrixQL();
  
  int cnt = 0;
  for(int i=0; i<2; i++){
    for(int j=0; j<2; j++){
      m.setMatrixQL()(i, j) = cnt++;
    }
  }

  cur_data = m.getMatrixQL();
  cout << cur_data << endl;
  printf("hello eigen \n");
}