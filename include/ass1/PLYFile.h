//
// Created by bsespede on 3/15/20.
//

#ifndef ROBOT_VISION_SRC_ASS1_POINTCLOUD_H_
#define ROBOT_VISION_SRC_ASS1_POINTCLOUD_H_

#include "ass1/CameraCalib.h"
#include <iostream>
#include <fstream>

class PLYFile {
 public:
  PLYFile(cv::Mat vertexColors, cv::Mat vertexPositions);
  void write(std::string scenePath, std::string filename);

 private:
  cv::Mat _vertexColor;
  cv::Mat _vertexPosition;
};

#endif //ROBOT_VISION_SRC_ASS1_POINTCLOUD_H_
