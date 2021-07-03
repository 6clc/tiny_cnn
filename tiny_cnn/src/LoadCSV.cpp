#include "LoadCSV.h"

namespace tinyDNN
{
  //����MNIST���ݼ���ѵ������55000��
  std::shared_ptr<Inter_LayerQL<double>> LoadCSV::input_Layer = std::make_shared<Inter_LayerQL<double>>(55000, 784);
  std::shared_ptr<Inter_LayerQL<double>> LoadCSV::output_Layer = std::make_shared<Inter_LayerQL<double>>(55000, 10);
  //����MNIST���ݼ������Լ���10000��
  std::shared_ptr<Inter_LayerQL<double>> LoadCSV::input_Layer_T = std::make_shared<Inter_LayerQL<double>>(10000, 784);
  std::shared_ptr<Inter_LayerQL<double>> LoadCSV::output_Layer_T = std::make_shared<Inter_LayerQL<double>>(10000, 10);
  //����MNIST���ݼ�����ά����ѵ������55000��
  std::vector<std::shared_ptr<MatrixQL<double>>> LoadCSV::conv_Input_Vector;
  //����MNIST���ݼ�����ά����
  std::vector<std::shared_ptr<MatrixQL<double>>> LoadCSV::conv_Input_Vector_T;
  //**********************************************************************************************************************
  //**********************************************************************************************************************
  //����Cifar���ݼ���ѵ������50000��
  std::vector<std::vector<std::shared_ptr<MatrixQL<double>>>> LoadCifar_10::cifar_Input_Vector;
  std::shared_ptr<MatrixQL<double>> LoadCifar_10::cifar_Out_Lable = std::make_shared<MatrixQL<double>>(50000, 10);

  std::vector<std::vector<std::shared_ptr<MatrixQL<double>>>> LoadCifar_10::cifar_Input_Vector_T;
  std::shared_ptr<MatrixQL<double>> LoadCifar_10::cifar_Out_Lable_T = std::make_shared<MatrixQL<double>>(10000, 10);

  //**********************************************************************************************************************

  LoadCSV::LoadCSV() {}

  LoadCSV::~LoadCSV() {}

  void LoadCSV::loadCSVTrain()
  {
    // ���� ѵ���� �� ѵ���ļ�
    std::ifstream inFile("H:/CNN_0510/DATA/MNISTDATA_CSV/train.csv", std::ios::in);
    std::string lineStr;
    int lineNum = 0;
    while (std::getline(inFile, lineStr))
    {
      std::stringstream ss(lineStr);
      std::string str;
      // ���ն��ŷָ�
      int inNum = 0;
      while (std::getline(ss, str, ','))
      {
        LoadCSV::input_Layer->forward_Matrix->setMatrixQL()(lineNum, inNum) = atoi(str.c_str());
        inNum++;
      }

      lineNum++;
    }
    //getchar();

    // ���� ѵ���� ��ѵ�� Lable�ļ�
    std::ifstream inFile_L("H:/CNN_0510/DATA/MNISTDATA_CSV/trainL.csv", std::ios::in);
    std::string lineStr_L;
    int lineNum_L = 0;
    while (std::getline(inFile_L, lineStr_L))
    {
      std::stringstream ss(lineStr_L);
      std::string str;
      // ���ն��ŷָ�
      int inNum = 0;
      while (std::getline(ss, str, ','))
      {
        LoadCSV::output_Layer->backward_Matrix->setMatrixQL()(lineNum_L, inNum) = atoi(str.c_str());
        inNum++;
      }
      lineNum_L++;
    }

    LoadCSV::input_Layer->forward_Matrix->setMatrixQL() = LoadCSV::input_Layer->forward_Matrix->getMatrixQL() / 255;

    //std::cout << MatrixWAndB::maTrixTrainToal.row(0) << std::endl;
    //std::cout << MatrixWAndB::maTrixTrainToalL.row(0) << std::endl;
  }

  void LoadCSV::loadCSVTest()
  {
    LoadCSV::input_Layer_T->forward_Matrix->setMatrixQL();
    LoadCSV::output_Layer_T->backward_Matrix->setMatrixQL();

    // ������Լ���ѵ���ļ�
    std::ifstream inFile("H:/CNN_0510/DATA/MNISTDATA_CSV/test.csv", std::ios::in);
    std::string lineStr;
    int lineNum = 0;
    while (std::getline(inFile, lineStr))
    {
      std::stringstream ss(lineStr);
      std::string str;
      // ���ն��ŷָ�
      int inNum = 0;
      while (std::getline(ss, str, ','))
      {
        LoadCSV::input_Layer_T->forward_Matrix->setMatrixQL()(lineNum, inNum) = atoi(str.c_str());
        inNum++;
      }

      lineNum++;
    }
    //getchar();

    // ������Լ���Lable�ļ�
    std::ifstream inFile_L("H:/CNN_0510/DATA/MNISTDATA_CSV/testL.csv", std::ios::in);
    std::string lineStr_L;
    int lineNum_L = 0;
    while (std::getline(inFile_L, lineStr_L))
    {
      std::stringstream ss(lineStr_L);
      std::string str;
      // ���ն��ŷָ�
      int inNum = 0;
      while (std::getline(ss, str, ','))
      {
        LoadCSV::output_Layer_T->backward_Matrix->setMatrixQL()(lineNum_L, inNum) = atoi(str.c_str());
        inNum++;
      }
      lineNum_L++;
    }

    LoadCSV::input_Layer_T->forward_Matrix->setMatrixQL() = LoadCSV::input_Layer_T->forward_Matrix->getMatrixQL() / 255;

    //std::cout << MatrixWAndB::maTrixTrainToal.row(0) << std::endl;
    //std::cout << MatrixWAndB::maTrixTrainToalL.row(0) << std::endl;
  }

  //��ѵ����ͼƬת��ΪVectorͼƬ����
  void LoadCSV::loadCSV_Train_Vector()
  {
    for (int i = 0; i < 55000; i++)
    {
      MatrixD trans_01 = static_cast<MatrixD>(LoadCSV::input_Layer->forward_Matrix->getMatrixQL().row(i));

      Eigen::Map<MatrixD> mapMatrix(trans_01.data(), 28, 28);

      std::shared_ptr<MatrixQL<double>> convMatrix = std::make_shared<MatrixQL<double>>(28, 28);
      convMatrix->setMatrixQL() = mapMatrix;

      conv_Input_Vector.push_back(convMatrix);
    }

    //std::cout << ((conv_Input_Vector[999])->getMatrixQL() * 9 ).cast<int>() << std::endl;
  }

  //�����Լ�ͼƬת��ΪVectorͼƬ����
  void LoadCSV::loadCSV_Test_Vector()
  {
    for (int i = 0; i < 10000; i++)
    {
      MatrixD trans_01 = static_cast<MatrixD>(LoadCSV::input_Layer_T->forward_Matrix->getMatrixQL().row(i));

      Eigen::Map<MatrixD> mapMatrix(trans_01.data(), 28, 28);

      std::shared_ptr<MatrixQL<double>> convMatrix = std::make_shared<MatrixQL<double>>(28, 28);
      convMatrix->setMatrixQL() = mapMatrix;

      conv_Input_Vector_T.push_back(convMatrix);
    }

    //std::cout << (( conv_Input_Vector_T[999] )->getMatrixQL() * 9).cast<int>() << std::endl;
  }

  //*******************************************************************************************************************************************************************
  void LoadCifar_10::loadCifar_10_Train()
  {
    LoadCifar_10::cifar_Out_Lable->setMatrixQL().setZero();
    LoadCifar_10::cifar_Out_Lable_T->setMatrixQL().setZero();

    for (int i = 1; i < 7; i++)
    {
      // ���� ѵ���� �� ѵ���ļ�
      std::string inHandler_Begin = "H:/tmp/cifar10_data/cifar-10-batches-bin/data_batch_";
      std::string inHandler_End = ".bin";

      std::string inHandler;

      switch (i)
      {
      case 6:
        inHandler = "H:/tmp/cifar10_data/cifar-10-batches-bin/test_batch.bin";
        break;
      default:
        inHandler = inHandler_Begin.append(std::to_string(i)).append(inHandler_End);
        break;
      }
      //std::string inHandler = inHandler_Begin.append(std::to_string(i)).append(inHandler_End);

      std::ifstream inFile(inHandler, std::ios::binary);
      char *buffer = new char[10000 * 3073];
      inFile.read(buffer, 10000 * 3073 * sizeof(char));

      for (int j = 0; j < 10000; j++)
      {
        unsigned char tmp = (unsigned char)buffer[j * 3073];
        //std::cout << (unsigned short)tmp << std::endl;

        switch (i)
        {
        case 6:
          LoadCifar_10::cifar_Out_Lable_T->setMatrixQL()(0 * 10000 + j, tmp) = 1;
          break;
        default:
          LoadCifar_10::cifar_Out_Lable->setMatrixQL()((i - 1) * 10000 + j, tmp) = 1;
          break;
        }
        //LoadCifar_10::cifar_Out_Lable->setMatrixQL()((i - 1) * 10000 + j, tmp) = 1;

        //����ͼƬ����������ͼƬ
        std::vector<std::shared_ptr<MatrixQL<double>>> inMatrix;

        for (int k = 0; k < 3; k++)
        {
          unsigned char tmp_01 = (unsigned char)buffer[j * 3073 + 1 + k * 1024];

          std::shared_ptr<MatrixQL<double>> mat = std::make_shared<MatrixQL<double>>(32, 32);
          for (int m = 0; m < 32; m++)
          {
            unsigned char tmp_02 = (unsigned char)buffer[j * 3073 + 1 + k * 1024 + m * 32];

            for (int n = 0; n < 32; n++)
            {
              unsigned char tmp_03 = (unsigned char)buffer[j * 3073 + 1 + k * 1024 + m * 32 + n];
              mat->setMatrixQL()(m, n) = tmp_03 / 255.0;
            }
          }
          //���ݾ�һ��
          double meanNum = mat->getMatrixQL().mean();
          mat->setMatrixQL() = mat->getMatrixQL().array() - meanNum;
          double stD = sqrt((((mat->getMatrixQL().array()).square()).sum() / (32 * 32)));
          mat->setMatrixQL() = mat->getMatrixQL() / stD;
          //std::cout << mat->getMatrixQL() << std::endl;

          inMatrix.push_back(mat);
        }
        switch (i)
        {
        case 6:
          LoadCifar_10::cifar_Input_Vector_T.push_back(inMatrix);
          break;
        default:
          LoadCifar_10::cifar_Input_Vector.push_back(inMatrix);
          break;
        }
      }
      //for (int j = 0 * 3073; j < 10000 * 3073; j++)
      //{
      //	unsigned char tmp = (unsigned char)buffer[j];
      //	std::cout << (unsigned short)tmp << std::endl;
      //}
      //std::cout << (unsigned short)(unsigned char)(buffer[3073 * 9999]) << std::endl;
      //std::cout << (unsigned short)(unsigned char)(buffer[3073 * 10000]) << std::endl;
    }
    //std::cout << cifar_Out_Lable->getMatrixQL() << std::endl;

    //std::cout << LoadCifar_10::cifar_Input_Vector[300][2]->getMatrixQL() << std::endl;
    //std::cout << LoadCifar_10::cifar_Input_Vector_T[300][2]->getMatrixQL() << std::endl;
    //std::cout << LoadCifar_10::cifar_Input_Vector.size() << std::endl;
    //std::cout << LoadCifar_10::cifar_Input_Vector_T.size() << std::endl;
    //std::cout << LoadCifar_10::cifar_Out_Lable_T->getMatrixQL() << std::endl;
    //std::cout << LoadCifar_10::cifar_Out_Lable->getMatrixQL() << std::endl;
  }

  //******************************************************************
  //����Tuli���ݼ���ѵ������8+8+8��
  std::vector<std::shared_ptr<MatrixQL<double>>> LoadTuLi::tuli_Train;
  std::vector<std::shared_ptr<MatrixQL<double>>> LoadTuLi::tuli_Test;

  void LoadTuLi::load_Tuli()
  {
    //	Train
    for (int i = 1; i < 24; i++)
    {
      // ���� ѵ���� �� ѵ���ļ�
      //	H:\CNN_0510\DATA\JZT\train
      std::string inHandler_Begin = "H:/CNN_0510/DATA/JZT/train/";
      std::string inHandler_End = ".txt";

      std::string inHandler;

      inHandler = inHandler_Begin.append(std::to_string(i)).append(inHandler_End);

      // ���� ѵ���� �� ѵ���ļ�
      std::ifstream inFile(inHandler, std::ios::in);
      std::string lineStr;
      int lineNum = 0;

      std::shared_ptr<MatrixQL<double>> mat = std::make_shared<MatrixQL<double>>(128, 128);
      std::cout << "i::" << i << std::endl;
      while (std::getline(inFile, lineStr))
      {
        std::stringstream ss(lineStr);
        std::string str;
        // ���ն��ŷָ�

        int inNum = 0;
        while (std::getline(ss, str, ' '))
        {
          //LoadCSV::input_Layer->forward_Matrix->setMatrixQL()(lineNum, inNum) = atoi(str.c_str());

          int row = inNum / 128;
          int col = inNum % 128;

          mat->setMatrixQL()(row, col) = atof(str.c_str());

          inNum++;
          //std::cout << inNum << std::endl;
        }

        lineNum++;
        std::cout << lineNum << std::endl;
      }

      std::cout << "First::" << mat->getMatrixQL()(0, 0) << std::endl;
      std::cout << "Last::" << mat->getMatrixQL()(127, 127) << std::endl;

      LoadTuLi::tuli_Train.push_back(mat);
    }
  }

  void LoadTuLi::load_Tuli_T()
  {
    //	Test
    for (int i = 1; i < 10; i++)
    {
      // ���� ѵ���� �� ѵ���ļ�
      //	H:\CNN_0510\DATA\JZT\train
      std::string inHandler_Begin = "H:/CNN_0510/DATA/JZT/test/";
      std::string inHandler_End = ".txt";

      std::string inHandler;

      inHandler = inHandler_Begin.append(std::to_string(i)).append(inHandler_End);

      // ���� ѵ���� �� ѵ���ļ�
      std::ifstream inFile(inHandler, std::ios::in);
      std::string lineStr;
      int lineNum = 0;

      std::shared_ptr<MatrixQL<double>> mat = std::make_shared<MatrixQL<double>>(128, 128);
      std::cout << "i::" << i << std::endl;
      while (std::getline(inFile, lineStr))
      {
        std::stringstream ss(lineStr);
        std::string str;
        // ���ն��ŷָ�

        int inNum = 0;
        while (std::getline(ss, str, ' '))
        {
          //LoadCSV::input_Layer->forward_Matrix->setMatrixQL()(lineNum, inNum) = atoi(str.c_str());

          int row = inNum / 128;
          int col = inNum % 128;

          mat->setMatrixQL()(row, col) = atof(str.c_str());

          inNum++;
          //std::cout << inNum << std::endl;
        }

        lineNum++;
        std::cout << lineNum << std::endl;
      }

      std::cout << "First::" << mat->getMatrixQL()(0, 0) << std::endl;
      std::cout << "Last::" << mat->getMatrixQL()(127, 127) << std::endl;

      LoadTuLi::tuli_Test.push_back(mat);
    }
  }

}