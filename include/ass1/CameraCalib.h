//
// Created by bsespede on 3/14/20.
//

#ifndef ROBOT_VISION_SRC_ASS1_CAMERACALIB_H_
#define ROBOT_VISION_SRC_ASS1_CAMERACALIB_H_

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <vector>
#include "ass1/ImageUtils.h"

class CameraCalib {
 public:
  struct Intrinsics {
    cv::Mat cameraMatrix;
    cv::Mat distCoeffs;
    cv::Size imageSize;
    float rmse;
  };
  struct ImagePoints {
    std::vector<cv::Point2f> points;
    std::string filename;
  };
  struct ObjectPoints {
    std::vector<cv::Point3f> points;
  };

  CameraCalib(std::string inputPath, cv::Size patternSize, float squareSize);
  void computeCalibration();
  Intrinsics getIntrinsics();
  void setIntrinsics(Intrinsics intrinsics);
  std::vector<ImagePoints> getImagePoints();
  std::vector<ObjectPoints> getObjectPoints();
  void printCalibration();

 private:
  void computeCornerPoints();

  std::string _inputPath;
  cv::Size _patternSize;
  float _squareSize;
  bool _hasCalibrated;

  std::vector<ImagePoints> _imagePoints;
  std::vector<ObjectPoints> _objectPoints;
  Intrinsics _intrinsics;
};

#endif //ROBOT_VISION_SRC_ASS1_CAMERACALIB_H_
