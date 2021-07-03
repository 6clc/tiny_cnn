#pragma once
#include <Eigen/Core>
#include <cmath>

namespace tinyDNN
{
  template <typename Dtype>
  using MatrixData = typename Eigen::Matrix<Dtype, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  template <typename Dtype>
  class MatrixQL{
    public :
      MatrixQL(int rowNum, int colNum);
      ~MatrixQL();

      const MatrixData<Dtype> &getMatrixQL() const;
      MatrixData<Dtype> &setMatrixQL();

    private:
      MatrixData<Dtype> matrixData;
      int rowNum;
      int colNum;
  };

template <typename Dtype>
MatrixQL<Dtype>::MatrixQL(int rowNum, int colNum) : rowNum(rowNum), colNum(colNum)
{
  this->matrixData.resize(this->rowNum, this->colNum);
}

template <typename Dtype>
MatrixQL<Dtype>::~MatrixQL()
{
}

template <typename Dtype>
inline const MatrixData<Dtype> &MatrixQL<Dtype>::getMatrixQL() const
{
  return this->matrixData;
}

template <typename Dtype>
inline MatrixData<Dtype> &MatrixQL<Dtype>::setMatrixQL()
{
  return this->matrixData;
}
}