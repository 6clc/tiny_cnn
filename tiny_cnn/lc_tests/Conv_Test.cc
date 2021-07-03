#include "../Bias_LayerQL.h"
#include "../Bias_Conv_Layer.h"
using namespace tinyDNN;

int main() {
  // Bias_LayerQL<float> biasLayerQL(LayerType::Bias_Layer, 3, 3, 1.);
  Bias_Conv_Layer<float> biasConvLayer(LayerType::Bias_Conv_Layer_L, 3, 3, 3);
  return 0;
}