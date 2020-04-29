//
// Created by bsespede on 3/14/20.
//

#include "ass1/CameraCalib.h"

CameraCalib::CameraCalib(std::string inputPath, cv::Size patternSize, float squareSize) : _inputPath(inputPath),
_patternSize(patternSize), _squareSize(squareSize), _hasCalibrated(false) {
  if (!boost::filesystem::is_directory(_inputPath)) {
    throw std::runtime_error("Input folder does not exist");
  }
}

void CameraCalib::computeCalibration() {
  printf("[DEBUG] Computing input and object points\n");
  computeCornerPoints();
  cv::Size imageSize = ImageUtils::getImagesSize(_inputPath);
  std::vector<cv::Mat> rvecs = std::vector<cv::Mat>(_imagePoints.size());
  std::vector<cv::Mat> tvecs = std::vector<cv::Mat>(_imagePoints.size());
  cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
  cv::Mat distCoeffs = cv::Mat::zeros(8, 1, CV_64F);

  printf("[DEBUG] Computing camera calibration\n");
  std::vector<std::vector<cv::Point2f>> imagePointsUnraveled;
  std::vector<std::vector<cv::Point3f>> objectPointsUnraveled;
  for (int imageNumber = 0; imageNumber < _imagePoints.size(); imageNumber++) {
    imagePointsUnraveled.push_back(_imagePoints[imageNumber].points);
    objectPointsUnraveled.push_back(_objectPoints[imageNumber].points);
  }

  float rmse = cv::calibrateCamera(objectPointsUnraveled, imagePointsUnraveled, imageSize, cameraMatrix, distCoeffs,
      rvecs, tvecs);

  _intrinsics = {cameraMatrix, distCoeffs, imageSize, rmse};
  _hasCalibrated = true;
}

void CameraCalib::computeCornerPoints() {
  printf("[DEBUG] Computing corner points\n");
  _imagePoints = std::vector<ImagePoints>();
  _objectPoints = std::vector<ObjectPoints>();

  std::vector<cv::Point3f> curObjectPoints;
  for (int i = 0; i < _patternSize.height; ++i) {
    for (int j = 0; j < _patternSize.width; ++j) {
      curObjectPoints.push_back(cv::Point3f(j * _squareSize, i * _squareSize, 0));
    }
  }

  std::vector<std::string> imagesPath = ImageUtils::getImagesPath(_inputPath);
  for (std::string imagePath : imagesPath) {
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    std::vector<cv::Point2f> curImagePoints;
    int flags = cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE;
    bool foundCorners = cv::findChessboardCorners(image, _patternSize, curImagePoints, flags);
    if (foundCorners) {
      cornerSubPix(image, curImagePoints, cv::Size(11, 11), cv::Size(-1, -1),
                   cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.01));
      _imagePoints.push_back({curImagePoints, boost::filesystem::path(imagePath).stem().string()});
      _objectPoints.push_back({curObjectPoints});
    }
  }
}


std::vector<CameraCalib::ImagePoints> CameraCalib::getImagePoints() {
  if (!_hasCalibrated) {
    throw std::runtime_error("Calibration has not been computed yet");
  }
  return _imagePoints;
}

std::vector<CameraCalib::ObjectPoints> CameraCalib::getObjectPoints() {
  if (!_hasCalibrated) {
    throw std::runtime_error("Calibration has not been computed yet");
  }
  return _objectPoints;
}

CameraCalib::Intrinsics CameraCalib::getIntrinsics() {
  if (!_hasCalibrated) {
    throw std::runtime_error("Calibration has not been computed yet");
  }
  return _intrinsics;
}

void CameraCalib::setIntrinsics(Intrinsics intrinsics) {
  _intrinsics = intrinsics;
}

void CameraCalib::printCalibration() {
  if (!_hasCalibrated) {
    throw std::runtime_error("Calibration has not been computed yet");
  }

  printf("[DEBUG] Intrinsics for \"%s\":\n", _inputPath.c_str());
  printf("fx: %f\n", _intrinsics.cameraMatrix.at<double>(0, 0));
  printf("fy: %f\n", _intrinsics.cameraMatrix.at<double>(1, 1));
  printf("cx: %f\n", _intrinsics.cameraMatrix.at<double>(0, 2));
  printf("cx: %f\n", _intrinsics.cameraMatrix.at<double>(1, 2));
  printf("k1: %f\n", _intrinsics.distCoeffs.at<double>(0, 0));
  printf("k2: %f\n", _intrinsics.distCoeffs.at<double>(1, 0));
  printf("p1: %f\n", _intrinsics.distCoeffs.at<double>(2, 0));
  printf("p2: %f\n", _intrinsics.distCoeffs.at<double>(3, 0));
  printf("k3: %f\n", _intrinsics.distCoeffs.at<double>(4, 0));
  printf("rmse: %f\n", _intrinsics.rmse);
}
