//
// Created by bsespede on 3/15/20.
//

#ifndef ROBOT_VISION_SRC_ASS1_STEREOCALIB_H_
#define ROBOT_VISION_SRC_ASS1_STEREOCALIB_H_

#include "ass1/CameraCalib.h"

class StereoCalib {
 public:
  struct Extrinsics {
    cv::Mat rotationMatrix;
    cv::Mat transVector;
    float rmse;
  };

  StereoCalib(std::string inputPath, cv::Size patternSize, float squareSize);
  void computeCalibration();
  CameraCalib::Intrinsics getLeftIntrinsics();
  CameraCalib::Intrinsics getRightIntrinsics();
  Extrinsics getExtrinsics();
  bool hasCalibrated();
  void printCalibration();

 private:
  std::string _inputPath;
  cv::Size _patternSize;
  float _squareSize;
  bool _hasCalibrated;

  CameraCalib _leftCalibration;
  CameraCalib _rightCalibration;
  Extrinsics _extrinsics;
};

#endif //ROBOT_VISION_SRC_ASS1_STEREOCALIB_H_
