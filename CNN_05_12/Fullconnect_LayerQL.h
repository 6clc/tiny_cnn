#pragma once
#include "LayerQL.h"
#include <random>

namespace tinyDNN
{
  template <typename Dtype>
  class Fullconnect_LayerQL : public LayerQL<Dtype>
  {
  public:
    friend class Test;
    Fullconnect_LayerQL(LayerType type, int rowNum, int colNum);
    ~Fullconnect_LayerQL() override final;

    void calForward(int type = 0) const override final;
    void calBackward(int type = 0) override final;

    void upMatrix() override final;
    void upMatrix_batch(Dtype upRate) override final;

    //static double upRate

  private:
    std::unique_ptr<MatrixQL<Dtype>> w_MatrixQL;
    int rowNum;
    int colNum;
  };

  //**********************************************************************************************
  template <typename Dtype>
  Fullconnect_LayerQL<Dtype>::Fullconnect_LayerQL(LayerType type, int rowNum, int colNum) : LayerQL<Dtype>(type)
  {
    std::cout << "Fullconnect_LayerQL Start!" << std::endl;

    this->rowNum = rowNum;
    this->colNum = colNum;
    this->w_MatrixQL = std::make_unique<MatrixQL<Dtype>>(rowNum, colNum);

    //this->w_MatrixQL->setMatrixQL().setRandom();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<Dtype> normal(0, 0.1);
    for (int i = 0; i < rowNum; i++)
    {
      for (int j = 0; j < colNum; j++)
      {
        this->w_MatrixQL->setMatrixQL()(i, j) = normal(gen);
      }
    }
  }

  template <typename Dtype>
  Fullconnect_LayerQL<Dtype>::~Fullconnect_LayerQL()
  {
    std::cout << "Fullconnect_LayerQL Over!" << std::endl;
  }

  template <typename Dtype>
  void Fullconnect_LayerQL<Dtype>::calForward(int type) const
  {
    //std::cout << this->w_MatrixQL->getMatrixQL() << std::endl;

    this->right_Layer->forward_Matrix->setMatrixQL() = this->left_Layer->forward_Matrix->getMatrixQL() * this->w_MatrixQL->getMatrixQL();
  }

  template <typename Dtype>
  void Fullconnect_LayerQL<Dtype>::calBackward(int type)
  {
    //std::cout << this->w_MatrixQL->getMatrixQL() << std::endl;

    //��	�������� * W.transpose
    this->left_Layer->backward_Matrix->setMatrixQL() = this->right_Layer->backward_Matrix->getMatrixQL() * this->w_MatrixQL->getMatrixQL().transpose();
  }

  template <typename Dtype>
  void Fullconnect_LayerQL<Dtype>::upMatrix()
  {
    //std::cout << this->w_MatrixQL->getMatrixQL() << std::endl;

    //����	ע��ת�� * ��������
    //this->w_MatrixQL->setMatrixQL() = (1-0)*(this->w_MatrixQL->getMatrixQL()) - 0.15 * (this->left_Layer->forward_Matrix->getMatrixQL().transpose() ) * ( this->right_Layer->backward_Matrix->getMatrixQL() );

    this->w_MatrixQL->setMatrixQL() = (1 - 0) * (this->w_MatrixQL->getMatrixQL()) - this->upFull * (this->left_Layer->forward_Matrix->getMatrixQL().transpose()) * (this->right_Layer->backward_Matrix->getMatrixQL());
  }

  template <typename Dtype>
  void Fullconnect_LayerQL<Dtype>::upMatrix_batch(Dtype upRate)
  {
    //std::cout << this->w_MatrixQL->getMatrixQL() << std::endl;

    //���ﲻ̫�ã����ڸ�һ��
    int rowNum_In = this->left_Layer->forward_Matrix->getMatrixQL().rows();
    std::unique_ptr<MatrixQL<Dtype>> oMatrix = std::make_unique<MatrixQL<Dtype>>(this->rowNum, this->colNum);
    oMatrix->setMatrixQL().setZero();

    //ÿһ������ת�� * ��������
    for (int i = 0; i < rowNum_In; i++)
    {
      oMatrix->setMatrixQL() = oMatrix->getMatrixQL() + (this->left_Layer->forward_Matrix->getMatrixQL().row(i).transpose()) * (this->right_Layer->backward_Matrix->getMatrixQL().row(i));
    }

    this->w_MatrixQL->setMatrixQL() = (1 - 0) * (this->w_MatrixQL->getMatrixQL()) - upRate * oMatrix->getMatrixQL();

    //std::cout << this->w_MatrixQL->getMatrixQL() << std::endl;
  }
}