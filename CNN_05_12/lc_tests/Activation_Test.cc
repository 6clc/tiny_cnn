#include "../Sigmoid_LayerQL.h"
#include "../SoftMax_LayerQL.h"
#include "../Relu_LayerQL.h"
#include "../PooLayerQL.h"
using namespace tinyDNN;

int main(){
  //Sigmoid_LayerQL<float> sigmoid(LayerType::Sigmoid_Layer);
  //SoftMax_LayerQL<float> softmax(LayerType::SoftMax_Layer);
  //Relu_LayerQL<float> relu(LayerType::Relu_Layer);
  PooLayerQL<float> pool(LayerType::Sigmoid_Layer, 0, 0);
  return 0;
}