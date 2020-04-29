//
// Created by bsespede on 3/17/20.
//

#include "ass1/ImageUtils.h"

cv::Size ImageUtils::getImagesSize(std::string inputPath) {
  bool sizeHasBeenSet = false;
  cv::Size imageSize;
  std::vector<std::string> imagesPath = getImagesPath(inputPath);
  for (std::string imagePath : imagesPath) {
    cv::Mat image = cv::imread(imagePath);
    if (!sizeHasBeenSet) {
      imageSize = image.size();
      sizeHasBeenSet = true;
      continue;
    }
    if (imageSize != image.size()) {
      throw std::runtime_error("Inconsistent image sizes");
    }
  }

  return imageSize;
}

std::vector<std::string> ImageUtils::getImagesPath(std::string inputPath) {
  std::vector<std::string> imagesPath;
  if (boost::filesystem::is_directory(inputPath)) {
    for (auto& file : boost::filesystem::directory_iterator(inputPath)) {
      if (file.path().extension() == ".png") {
        imagesPath.push_back(file.path().string());
      }
    }
  }
  else {
    throw std::runtime_error("Input folder doesn't exist");
  }
  if (imagesPath.empty()) {
    throw std::runtime_error("No images found in folder");
  }
  return imagesPath;
}