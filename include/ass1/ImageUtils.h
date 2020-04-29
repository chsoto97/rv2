//
// Created by bsespede on 3/17/20.
//

#ifndef ROBOT_VISION_INCLUDE_ASS1_IMAGEUTILS_H_
#define ROBOT_VISION_INCLUDE_ASS1_IMAGEUTILS_H_

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <boost/filesystem.hpp>
#include <vector>
#include <string>
#include <exception>

namespace ImageUtils {
  std::vector<std::string> getImagesPath(std::string imagesPath);
  cv::Size getImagesSize(std::string inputPath);
};

#endif //ROBOT_VISION_INCLUDE_ASS1_IMAGEUTILS_H_
