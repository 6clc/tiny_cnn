#include "Fullconnect_LayerQL.h"
using namespace tinyDNN;
int main(){
  Fullconnect_LayerQL<float> full_connect_layer(LayerType::Fullconnect_Layer, 3, 3);
  return 0;
}