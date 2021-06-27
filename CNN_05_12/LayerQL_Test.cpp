#include "LayerQL.h"
using namespace tinyDNN;
int main(){
  // 注释掉纯虚函数，才可以运行此处
      // virtual void calForward(int type = 0) const = 0;
    // virtual void calBackward(int type = 0) = 0;
    // virtual void upMatrix() = 0;

    // virtual void upMatrix_batch(Dtype upRate) = 0;

  LayerQL<double> layer(LayerType::Fullconnect_Layer);
  return 0;
}

// Question: not use  operator+ ????