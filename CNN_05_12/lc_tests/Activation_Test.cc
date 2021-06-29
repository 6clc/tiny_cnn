#include "../PooLayerQL.h"
#include "../PooLayerQL.cc" //模板类，不知道如何分开优化
#include "../Relu_LayerQL.h"
#include "../Sigmoid_LayerQL.h"
#include "../SoftMax_LayerQL.h"
#include "../Padding_LayerQL.h"
#include "../MSE_Loss_LayerQL.h"
#include "../LoadCSV.h"
using namespace tinyDNN;

int main(){
  //Sigmoid_LayerQL<float> sigmoid(LayerType::Sigmoid_Layer);
  //SoftMax_LayerQL<float> softmax(LayerType::SoftMax_Layer);
  //Relu_LayerQL<float> relu(LayerType::Relu_Layer);
  // PooLayerQL<float> pool(LayerType::Sigmoid_Layer, 0, 0);
  // Padding_LayerQL<float> padding(LayerType::Padding_Layer, 3, 3, 1);
 // MSE_Loss_LayerQL<float> mse(LayerType::MSE_Loss_Layer);
 LoadCSV loadCsv();
  return 0;
}