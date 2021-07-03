#include "Padding_LayerQL.h"
using namespace tinyDNN;

int main(){
  Padding_LayerQL<float> padding_layer(LayerType::Padding_Layer, 3, 3, 1);
  return 0;
}