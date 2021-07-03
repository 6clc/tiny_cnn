#pragma once
#include "LayerQL.h"
namespace tinyDNN
{
  template <typename Dtype>
  class Bias_LayerQL : public LayerQL<Dtype>
  {
  public:
    Bias_LayerQL(LayerType type, int rowNum, int colNum, Dtype ranNum);
    ~Bias_LayerQL() override final;

    void calForward(int type = 0) const override final;
    void calBackward(int type = 0) override final;
    void upMatrix() override final;
    void upMatrix_batch(Dtype upRate) override final;

  protected:
    std::unique_ptr<MatrixQL<Dtype>> b_MatrixQL;
  };
  //*******************************************************************************************************************************
  template <typename Dtype>
  Bias_LayerQL<Dtype>::Bias_LayerQL(LayerType type, int rowNum, int colNum, Dtype ranNum) : LayerQL<Dtype>(type)
  {
    std::cout << "Bias_LayerQL Start!" << std::endl;
    this->b_MatrixQL = std::make_unique<MatrixQL<Dtype>>(rowNum, colNum);
    //���������Ҫ��
    this->b_MatrixQL->setMatrixQL().setConstant(ranNum);
    //this->b_MatrixQL->setMatrixQL().setZero();
  }

  template <typename Dtype>
  Bias_LayerQL<Dtype>::~Bias_LayerQL()
  {
    std::cout << "Bias_LayerQL Over!" << std::endl;
  }

  template <typename Dtype>
  void Bias_LayerQL<Dtype>::calForward(int type ) const
  {
    //std::cout << this->b_MatrixQL->getMatrixQL() << std::endl;

    //����Ĭ��bֻ��һ�У����������b��һ����չ����ÿһ�ж�����һ��b
    int rowNum = this->left_Layer->forward_Matrix->getMatrixQL().rows();
    std::unique_ptr<MatrixQL<Dtype>> oMatrix = std::make_unique<MatrixQL<Dtype>>(rowNum, 1);
    oMatrix->setMatrixQL().setOnes();

    this->right_Layer->forward_Matrix->setMatrixQL() = this->left_Layer->forward_Matrix->getMatrixQL() + (oMatrix->getMatrixQL()) * (this->b_MatrixQL->getMatrixQL());

    //==========================================================================================

    ////����ǵ����������������,������SGD��ʱ��������һЩ
    //this->right_Layer->forward_Matrix->setMatrixQL() = this->left_Layer->forward_Matrix->getMatrixQL() + (this->b_MatrixQL->getMatrixQL());
  }

  template <typename Dtype>
  void Bias_LayerQL<Dtype>::calBackward(int type )
  {
    //std::cout << this->b_MatrixQL->getMatrixQL() << std::endl;

    //���򴫲���ֱ�ӹ�ȥ�ͺ�
    this->left_Layer->backward_Matrix->setMatrixQL() = this->right_Layer->backward_Matrix->getMatrixQL();
  }

  template <typename Dtype>
  void Bias_LayerQL<Dtype>::upMatrix()
  {
    //std::cout << this->right_Layer->backward_Matrix->getMatrixQL() << std::endl;

    //ƫ�ø��£�����ѧϰ������Ϊ0.5
    this->b_MatrixQL->setMatrixQL() = this->b_MatrixQL->getMatrixQL() - 0.5 * (this->right_Layer->backward_Matrix->getMatrixQL());

    //std::cout << this->b_MatrixQL->getMatrixQL() << std::endl;
  }

  template <typename Dtype>
  void Bias_LayerQL<Dtype>::upMatrix_batch(Dtype upRate)
  {
    //std::cout << this->b_MatrixQL->getMatrixQL() << std::endl;

    //ƫ�ø��£�������Ҫ�Է���İ������Ȼ�����ֵ
    this->b_MatrixQL->setMatrixQL() = this->b_MatrixQL->getMatrixQL() - 0.1 * this->right_Layer->backward_Matrix->getMatrixQL().colwise().sum();

    //std::cout << this->b_MatrixQL->getMatrixQL() << std::endl;
  }
}