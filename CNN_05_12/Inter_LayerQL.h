#pragma once
#include "MatrixQL.h"
#include "memory"
#include <iostream>
#include <vector>

namespace tinyDNN
{
  template <typename Dtype>
  class LayerQL;

  template <typename Dtype>
  class Inter_LayerQL
  {
  public:
    friend class Test;
    //friend calss Mnist_Conv_Test;
    template <typename T>
    friend std::shared_ptr<Inter_LayerQL<T>> operator+(std::shared_ptr<Inter_LayerQL<T>> &operLeft, std::shared_ptr<LayerQL<T>> &operRight);

    Inter_LayerQL(int rowNum=0, int colNum=0);
    ~Inter_LayerQL();

  public:
    std::shared_ptr<MatrixQL<Dtype>> forward_Matrix;
    std::shared_ptr<MatrixQL<Dtype>> backward_Matrix;

    std::vector<std::shared_ptr<MatrixQL<Dtype>>> forward_Matrix_Vector;
    std::vector<std::shared_ptr<MatrixQL<Dtype>>> backward_Matrix_Vector;
  };

  template <typename Dtype>
  Inter_LayerQL<Dtype>::Inter_LayerQL(int rowNum, int colNum)
  {
    std::cout << "Inter_Layer Start!" << std::endl;

    this->forward_Matrix = std::make_unique<MatrixQL<Dtype>>(rowNum, colNum);
    this->backward_Matrix = std::make_unique<MatrixQL<Dtype>>(rowNum, colNum);
  }

  template <typename Dtype>
  Inter_LayerQL<Dtype>::~Inter_LayerQL()
  {
    std::cout << "Inter_Layer End!" << std::endl;
  }
}
