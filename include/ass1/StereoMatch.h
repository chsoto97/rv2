//
// Created by bsespede on 3/15/20.
//

#ifndef ROBOT_VISION_SRC_ASS1_STEREOMATCH_H_
#define ROBOT_VISION_SRC_ASS1_STEREOMATCH_H_

#include "ass1/StereoCalib.h"

class StereoMatch {
 public:
  StereoMatch(std::string inputPath, StereoCalib calibration);
  void computeRectification();
  void computeDepth(bool storeDisparityMap, bool storeCloud);

 private:
  std::string _inputPath;
  cv::Mat _projectionMatrix;
  bool _hasRectified;

  StereoCalib _calibration;
};

#endif //ROBOT_VISION_SRC_ASS1_STEREOMATCH_H_
