#include "Inter_LayerQL.h"
using namespace tinyDNN;
int main() {
  Inter_LayerQL<int> layer(3, 3);
  layer.forward_Matrix;
  layer.forward_Matrix_Vector;
  layer.backward_Matrix;
  layer.backward_Matrix_Vector;
  return 0;
}