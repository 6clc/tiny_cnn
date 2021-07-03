#pragma once
#include "LayerQL.h"
#include <vector>

namespace tinyDNN
{
  template <typename Dtype>
  class Conv_Kernel
  {
  public:
    Conv_Kernel(int rowNum, int colNum, int kernelWidth, int kernelSize, int paddingSize) : rowNum(rowNum), colNum(colNum), kernelWidth(kernelWidth), kernelSize(kernelSize), paddingSize(paddingSize)
    {
      for (int i = 0; i < kernelSize; i++)
      {
        std::shared_ptr<MatrixQL<Dtype>> oneSlice_Kernel = std::make_shared<MatrixQL<Dtype>>(kernelWidth, kernelWidth);

        //����
        ////oneSlice_Kernel->setMatrixQL().setOnes();
        //double startNum = 0.1 * (i + 1) ;
        //for ( int p = 0; p < kernelWidth; p++ )
        //{
        //	for ( int q = 0; q < kernelWidth; q++ )
        //	{
        //		oneSlice_Kernel->setMatrixQL()(p, q) = startNum;
        //		startNum = startNum + 0.1 * (i + 1);
        //	}
        //}

        ////ʵ��
        //oneSlice_Kernel->setMatrixQL().setRandom();

        //ʵ�٣� ���ø�˹�������̬�ֲ�
        std::random_device rd;
        std::mt19937 gen(rd());
        //ƽ��ֵ1����׼�� 0.1
        std::normal_distribution<Dtype> normal(0, 0.01);
        for (int p = 0; p < kernelWidth; p++)
        {
          for (int q = 0; q < kernelWidth; q++)
          {
            oneSlice_Kernel->setMatrixQL()(p, q) = normal(gen);
          }
        }

        //	һ����������iƬ
        this->conv_Kernel_Vector.push_back(oneSlice_Kernel);
      }
    }
    //	�������Ƭ���Ͻ�����˲����
    void conv_CalForward(std::vector<std::shared_ptr<MatrixQL<Dtype>>> &inMatrixVector, std::shared_ptr<MatrixQL<Dtype>> &outMatrix)
    {

      for (int i = 0; i < kernelSize; i++)
      {
        //std::cout << inMatrixVector[i]->getMatrixQL() << std::endl;
        //std::cout << conv_Kernel_Vector[i]->getMatrixQL() << std::endl;

        ////�����������
        std::shared_ptr<MatrixQL<Dtype>> paddingMatrix = std::make_shared<MatrixQL<Dtype>>(rowNum + 2 * paddingSize, colNum + 2 * paddingSize);
        paddingMatrix->setMatrixQL().setZero();
        paddingMatrix->setMatrixQL().block(paddingSize, paddingSize, rowNum, colNum) = inMatrixVector[i]->getMatrixQL().block(0, 0, rowNum, colNum);

        //��ÿһ�������˽��м���
        outMatrix->setMatrixQL() = outMatrix->getMatrixQL() + conv_Matrix(paddingMatrix, conv_Kernel_Vector[i])->getMatrixQL();
      }
    }
    //��ÿһ��ͼ���о�������
    std::shared_ptr<MatrixQL<Dtype>> conv_Matrix(std::shared_ptr<MatrixQL<Dtype>> &inMatrixPtr, std::shared_ptr<MatrixQL<Dtype>> &convMatrixPtr)
    {
      std::shared_ptr<MatrixQL<Dtype>> reMatrix = std::make_shared<MatrixQL<Dtype>>(rowNum, colNum);
      for (int i = 0; i < rowNum; i++)
      {
        for (int j = 0; j < colNum; j++)
        {
          reMatrix->setMatrixQL()(i, j) = (inMatrixPtr->getMatrixQL().block(i, j, kernelWidth, kernelWidth).array() * convMatrixPtr->getMatrixQL().array()).sum();
        }
      }
      return reMatrix;
    }

  public:
    std::vector<std::shared_ptr<MatrixQL<Dtype>>> conv_Kernel_Vector;

    //���������к���
    int rowNum;
    int colNum;
    //�����˵Ŀ���
    int kernelWidth;
    //�����˵�Ƭ��
    int kernelSize;
    //�����˵�����
    int paddingSize;
  };

  //=======================================================================================================================

  template <typename Dtype>
  class Conv_LayerQL : public LayerQL<Dtype>
  {
  public:
    //							����			��������			����			����				�����˿���		�����˼�Ƭ		�������
    Conv_LayerQL(LayerType type, int kernelNum, int rowNum, int colNum, int kernelWidth, int kernelSize, int paddingSize) : LayerQL(type), kernelNum(kernelNum), rowNum(rowNum), colNum(colNum), kernelWidth(kernelWidth), kernelSize(kernelSize), paddingSize(paddingSize)
    {
      std::cout << "Conv_LayerQL Start!" << std::endl;

      for (int i = 0; i < kernelNum; i++)
      {
        std::shared_ptr<Conv_Kernel<Dtype>> oneKernel = std::make_shared<Conv_Kernel<Dtype>>(rowNum, colNum, kernelWidth, kernelSize, paddingSize);

        this->conv_Kernel_Vector.push_back(oneKernel);
      }
    }

    ~Conv_LayerQL() override final
    {
      std::cout << "Conv_LayerQL Over!" << std::endl;
    }

    void calForward(int type = 0) const override final
    {
      //ÿ����ǰ�����������������е����������²���
      this->right_Layer->forward_Matrix_Vector.clear();

      for (int i = 0; i < kernelNum; i++)
      {
        std::shared_ptr<MatrixQL<Dtype>> outMatrix = std::make_shared<MatrixQL<Dtype>>(rowNum, colNum);
        outMatrix->setMatrixQL().setZero();

        this->conv_Kernel_Vector[i]->conv_CalForward(this->left_Layer->forward_Matrix_Vector, outMatrix);
        this->right_Layer->forward_Matrix_Vector.push_back(outMatrix);
      }
    }

    void calBackward(int type = 0) override final
    {
      this->left_Layer->backward_Matrix_Vector.clear();

      for (int i = 0; i < kernelSize; i++)
      {
        std::shared_ptr<MatrixQL<Dtype>> matrix_Left = std::make_shared<MatrixQL<Dtype>>(rowNum, colNum);
        matrix_Left->setMatrixQL().setZero();
        this->left_Layer->backward_Matrix_Vector.push_back(matrix_Left);
      }

      for (int i = 0; i < kernelNum; i++)
      {
        std::shared_ptr<MatrixQL<Dtype>> paddingMatrix = std::make_shared<MatrixQL<Dtype>>(rowNum + 2 * paddingSize, colNum + 2 * paddingSize);
        paddingMatrix->setMatrixQL().setZero();
        paddingMatrix->setMatrixQL().block(paddingSize, paddingSize, rowNum, colNum) = this->right_Layer->backward_Matrix_Vector[i]->getMatrixQL().block(0, 0, rowNum, colNum);

        for (int j = 0; j < kernelSize; j++)
        {
          std::shared_ptr<MatrixQL<Dtype>> reMatrix = std::make_shared<MatrixQL<Dtype>>(rowNum, colNum);
          for (int p = 0; p < rowNum; p++)
          {
            for (int q = 0; q < colNum; q++)
            {
              reMatrix->setMatrixQL()(p, q) = (paddingMatrix->getMatrixQL().block(p, q, kernelWidth, kernelWidth).array() * conv_Kernel_Vector[i]->conv_Kernel_Vector[j]->getMatrixQL().reverse().array()).sum();
            }
          }
          this->left_Layer->backward_Matrix_Vector[j]->setMatrixQL() = this->left_Layer->backward_Matrix_Vector[j]->getMatrixQL() + reMatrix->getMatrixQL();
        }
      }
    };

    void upMatrix() override final
    {
      for (int i = 0; i < kernelNum; i++)
      {
        for (int j = 0; j < kernelSize; j++)
        {
          std::shared_ptr<MatrixQL<Dtype>> paddingMatrix = std::make_shared<MatrixQL<Dtype>>(rowNum + 2 * paddingSize, colNum + 2 * paddingSize);
          paddingMatrix->setMatrixQL().setZero();
          paddingMatrix->setMatrixQL().block(paddingSize, paddingSize, rowNum, colNum) = this->left_Layer->forward_Matrix_Vector[j]->getMatrixQL().block(0, 0, rowNum, colNum);

          std::shared_ptr<MatrixQL<Dtype>> upMatrix = std::make_shared<MatrixQL<Dtype>>(kernelWidth, kernelWidth);
          for (int p = 0; p < kernelWidth; p++)
          {
            for (int q = 0; q < kernelWidth; q++)
            {
              upMatrix->setMatrixQL()(p, q) = (paddingMatrix->getMatrixQL().block(p, q, rowNum, colNum).array() *
                                               this->right_Layer->backward_Matrix_Vector[i]->getMatrixQL().array())
                                                  .sum();
            }
          }

          //std::cout << "UPֵ++++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
          //std::cout << upMatrix->getMatrixQL() << std::endl;
          //std::cout << "��֮ǰ++++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
          //std::cout << this->conv_Kernel_Vector[i]->conv_Kernel_Vector[j]->getMatrixQL() << std::endl;

          //*********************************************
          //this->conv_Kernel_Vector[i]->conv_Kernel_Vector[j]->setMatrixQL() = this->conv_Kernel_Vector[i]->conv_Kernel_Vector[j]->getMatrixQL() - 0.5 * upMatrix->getMatrixQL();

          this->conv_Kernel_Vector[i]->conv_Kernel_Vector[j]->setMatrixQL() = this->conv_Kernel_Vector[i]->conv_Kernel_Vector[j]->getMatrixQL() - this->upConv * upMatrix->getMatrixQL();
          //*********************************************

          //std::cout << "��֮��++++++++++++++++++++++++++++++++++++++++++++++++++" << std::endl;
          //std::cout << this->conv_Kernel_Vector[i]->conv_Kernel_Vector[j]->getMatrixQL() << std::endl;
        }
      }
    };
    void upMatrix_batch(Dtype upRate) override final{};

  public:
    std::vector<std::shared_ptr<Conv_Kernel<Dtype>>> conv_Kernel_Vector;
    //�����˵ĸ���
    int kernelNum;
    //���������к���
    int rowNum;
    int colNum;
    //�����˵Ŀ���
    int kernelWidth;
    //�����˵�Ƭ��
    int kernelSize;
    //�����С
    int paddingSize;
  };
}