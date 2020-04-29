//
// Created by bsespede on 3/15/20.
//

#include "ass1/StereoCalib.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

StereoCalib::StereoCalib(std::string inputPath, cv::Size patternSize, float squareSize) : _inputPath(inputPath),
_patternSize(patternSize), _squareSize(squareSize), _hasCalibrated(false),
_leftCalibration(inputPath + "/left", patternSize, squareSize),
_rightCalibration(inputPath + "/right", patternSize, squareSize) {
  if (!boost::filesystem::is_directory(inputPath)) {
    throw std::runtime_error("Input folder does not exist");
  }
}

void StereoCalib::computeCalibration() {
  printf("[DEBUG] Computing intrinsics for left camera\n");
  _leftCalibration.computeCalibration();

  printf("[DEBUG] Computing intrinsics for right camera\n");
  _rightCalibration.computeCalibration();

  std::vector<std::vector<cv::Point2f>> leftImagePointsUnraveled;
  std::vector<std::vector<cv::Point2f>> rightImagePointsUnraveled;
  std::vector<std::vector<cv::Point3f>> objectPointsUnraveled;
  for (CameraCalib::ImagePoints leftImagePoints: _leftCalibration.getImagePoints()) {
    for (CameraCalib::ImagePoints rightImagePoints: _rightCalibration.getImagePoints()) {
      if (leftImagePoints.filename == rightImagePoints.filename) {
        leftImagePointsUnraveled.push_back(leftImagePoints.points);
        rightImagePointsUnraveled.push_back(rightImagePoints.points);
        objectPointsUnraveled.push_back(_leftCalibration.getObjectPoints()[0].points);
        break;
      }
    }
  }

  printf("[DEBUG] Computing extrinsics between cameras\n");
  cv::Mat leftCameraMatrix = _leftCalibration.getIntrinsics().cameraMatrix;
  cv::Mat rightCameraMatrix = _rightCalibration.getIntrinsics().cameraMatrix;
  cv::Mat leftDistortions = _leftCalibration.getIntrinsics().distCoeffs;
  cv::Mat rightDistortions = _rightCalibration.getIntrinsics().distCoeffs;
  cv::Size imageSize = _leftCalibration.getIntrinsics().imageSize;
  cv::Mat rotationMatrix;
  cv::Mat translationMatrix;
  cv::Mat essentialMatrix;
  cv::Mat fundamentalMatrix;
  float rmse = cv::stereoCalibrate(objectPointsUnraveled, leftImagePointsUnraveled, rightImagePointsUnraveled,
      leftCameraMatrix, leftDistortions, rightCameraMatrix, rightDistortions, imageSize, rotationMatrix,
      translationMatrix, essentialMatrix, fundamentalMatrix);

  _leftCalibration.setIntrinsics({leftCameraMatrix, leftDistortions, imageSize, _leftCalibration.getIntrinsics().rmse});
  _rightCalibration.setIntrinsics({rightCameraMatrix, rightDistortions, imageSize, _rightCalibration.getIntrinsics().rmse});
  _extrinsics = {rotationMatrix, translationMatrix, rmse};
  _hasCalibrated = true;
}

CameraCalib::Intrinsics StereoCalib::getLeftIntrinsics() {
  if (!_hasCalibrated) {
    throw std::runtime_error("Stereo calibration has not been computed yet");
  }
  return _leftCalibration.getIntrinsics();
}

CameraCalib::Intrinsics StereoCalib::getRightIntrinsics() {
  if (!_hasCalibrated) {
    throw std::runtime_error("Stereo calibration has not been computed yet");
  }
  return _rightCalibration.getIntrinsics();
}

StereoCalib::Extrinsics StereoCalib::getExtrinsics() {
  if (!_hasCalibrated) {
    throw std::runtime_error("Stereo calibration has not been computed yet");
  }
  return _extrinsics;
}

bool StereoCalib::hasCalibrated() {
  return _hasCalibrated;
}

void StereoCalib::printCalibration() {
  if (!_hasCalibrated) {
    throw std::runtime_error("Stereo calibration has not been computed yet");
  }

  _leftCalibration.printCalibration();
  _rightCalibration.printCalibration();

  printf("[DEBUG] Extrinsics for \"%s\":\n", _inputPath.c_str());
  printf("r00: %f\n", _extrinsics.rotationMatrix.at<double>(0, 0));
  printf("r01: %f\n", _extrinsics.rotationMatrix.at<double>(0, 1));
  printf("r02: %f\n", _extrinsics.rotationMatrix.at<double>(0, 2));
  printf("r10: %f\n", _extrinsics.rotationMatrix.at<double>(1, 0));
  printf("r11: %f\n", _extrinsics.rotationMatrix.at<double>(1, 1));
  printf("r12: %f\n", _extrinsics.rotationMatrix.at<double>(1, 2));
  printf("r20: %f\n", _extrinsics.rotationMatrix.at<double>(2, 0));
  printf("r21: %f\n", _extrinsics.rotationMatrix.at<double>(2, 1));
  printf("r22: %f\n", _extrinsics.rotationMatrix.at<double>(2, 2));
  printf("tx: %f\n", _extrinsics.transVector.at<double>(0, 0));
  printf("ty: %f\n", _extrinsics.transVector.at<double>(0, 1));
  printf("tz: %f\n", _extrinsics.transVector.at<double>(0, 2));
  printf("rmse: %f\n", _extrinsics.rmse);
}