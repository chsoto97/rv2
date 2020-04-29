//
// Created by bsespede on 3/15/20.
//

#include <ass1/PLYFile.h>
#include "ass1/StereoMatch.h"

StereoMatch::StereoMatch(std::string inputPath, StereoCalib calibration) : _inputPath(inputPath),
_calibration(calibration) {}

void StereoMatch::computeRectification() {
  if (!_calibration.hasCalibrated()) {
    throw std::runtime_error("Calibration has not been computed");
  }

  _calibration.printCalibration();

  std::string inputPathLeft = _inputPath + "/left";
  std::string inputPathRight = _inputPath + "/right";
  std::string outputPathLeft = inputPathLeft + "/rectified";
  std::string outputPathRight = inputPathRight + "/rectified";
  boost::filesystem::create_directories(outputPathLeft);
  boost::filesystem::create_directories(outputPathRight);

  cv::Mat leftRectification;
  cv::Mat rightRectification;
  cv::Mat leftProjection;
  cv::Mat rightProjection;
  cv::Mat disparityToDepth;
  CameraCalib::Intrinsics leftIntrinsics = _calibration.getLeftIntrinsics();
  CameraCalib::Intrinsics rightIntrinsics = _calibration.getRightIntrinsics();
  StereoCalib::Extrinsics extrinsics = _calibration.getExtrinsics();
  cv::Size imageSize = _calibration.getLeftIntrinsics().imageSize;

  printf("[DEBUG] Computing projection matrices\n");
  cv::stereoRectify(leftIntrinsics.cameraMatrix, leftIntrinsics.distCoeffs,
                    rightIntrinsics.cameraMatrix, rightIntrinsics.distCoeffs,
                    imageSize, extrinsics.rotationMatrix, extrinsics.transVector,
                    leftRectification, rightRectification, leftProjection, rightProjection, disparityToDepth);
  _projectionMatrix = disparityToDepth;

  printf("[DEBUG] Computing mapping matrices\n");
  cv::Mat leftMapX;
  cv::Mat leftMapY;
  cv::Mat rightMapX;
  cv::Mat rightMapY;
  cv::initUndistortRectifyMap(leftIntrinsics.cameraMatrix, leftIntrinsics.distCoeffs,
                              leftRectification, leftProjection, imageSize, CV_16SC2, leftMapX, leftMapY);
  cv::initUndistortRectifyMap(rightIntrinsics.cameraMatrix, rightIntrinsics.distCoeffs,
                              rightRectification, rightProjection, imageSize, CV_16SC2, rightMapX, rightMapY);

  printf("[DEBUG] Rectifying left camera images\n");
  std::vector<std::string> imagesPathLeft = ImageUtils::getImagesPath(inputPathLeft);
  for (std::string imagePath : imagesPathLeft) {
    cv::Mat leftImage = cv::imread(imagePath, cv::IMREAD_COLOR);
    cv::Mat leftImageUndistorted;
    cv::remap(leftImage, leftImageUndistorted, leftMapX, leftMapY, cv::INTER_LINEAR);
    std::string filename = boost::filesystem::path(imagePath).filename().string();
    cv::imwrite(outputPathLeft + "/" + filename, leftImageUndistorted);
  }

  printf("[DEBUG] Rectifying right camera images\n");
  std::vector<std::string> imagesPathRight = ImageUtils::getImagesPath(inputPathRight);
  for (std::string imagePath : imagesPathRight) {
    cv::Mat rightImage = cv::imread(imagePath, cv::IMREAD_COLOR);
    cv::Mat rightImageUndistorted;
    cv::remap(rightImage, rightImageUndistorted, rightMapX, rightMapY, cv::INTER_LINEAR);
    std::string filename = boost::filesystem::path(imagePath).filename().string();
    cv::imwrite(outputPathRight + "/" + filename, rightImageUndistorted);
  }
  _hasRectified=true;
}

void StereoMatch::computeDepth(bool storeDisparityMap, bool storeCloud) {
  if (!_hasRectified) {
    throw std::runtime_error("Rectification has not been computed");
  }

  std::string inputPathLeft = _inputPath + "/left/rectified";
  std::string inputPathRight = _inputPath + "/right/rectified";
  std::string outputPathDisparity = _inputPath + "/disparity";
  std::string outputPathCloud = _inputPath + "/cloud";
  boost::filesystem::create_directories(outputPathDisparity);
  boost::filesystem::create_directories(outputPathCloud);

  std::vector<std::string> imagesPathLeft = ImageUtils::getImagesPath(inputPathLeft);
  for (std::string leftImagePath : imagesPathLeft) {
    printf("[DEBUG] Computing disparity map and point cloud of \"%s\"\n", leftImagePath.c_str());
    std::string filename = boost::filesystem::path(leftImagePath).stem().string();
    std::string rightImagePath = inputPathRight + "/" + filename + ".png";
    cv::Mat leftImage = cv::imread(leftImagePath, cv::IMREAD_ANYCOLOR);
    cv::Mat rightImage = cv::imread(rightImagePath, cv::IMREAD_ANYCOLOR);
    cv::Mat disparityMap;

    int channels = 3;
    int windowSize = 3;
    int minDisparity = 0;
    int maxDisparity = 512;
    int P1 = 8 * channels * windowSize * windowSize;
    int P2 = 32 * channels * windowSize * windowSize;
    cv::Ptr<cv::StereoSGBM> stereoAlgorithm = cv::StereoSGBM::create(minDisparity, maxDisparity, windowSize, P1, P2, 0, 0, 0, 0, 0);
    stereoAlgorithm->setMode(cv::StereoSGBM::MODE_HH);
    stereoAlgorithm->compute(leftImage, rightImage, disparityMap);

    if (storeDisparityMap) {
      cv::Mat disparityInteger;
      disparityMap.convertTo(disparityInteger, CV_8U, 255 / ((maxDisparity - minDisparity) * 16.0f));
      cv::imwrite(outputPathDisparity + "/" + filename + ".png", disparityInteger);
    }

    if (storeCloud) {
      cv::Mat disparityFloat;
      cv::Mat pointCloud;
      disparityMap.convertTo(disparityFloat, CV_32F, 1.0f / 16.0f);
      cv::reprojectImageTo3D(disparityFloat, pointCloud, _projectionMatrix);
      PLYFile pointCloudFile(leftImage, pointCloud);
      pointCloudFile.write(outputPathCloud, filename);
    }
  }
}