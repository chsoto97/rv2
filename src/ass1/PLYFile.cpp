//
// Created by bsespede on 3/15/20.
//

#include "ass1/PLYFile.h"

PLYFile::PLYFile(cv::Mat vertexColor, cv::Mat vertexPosition) : _vertexColor(vertexColor), _vertexPosition(vertexPosition) {}

void PLYFile::write(std::string scenePath, std::string filename) {
  std::string pointCloudPath = scenePath + "/" + filename + ".ply";

  int totalValidPixels = 0;
  for (int u = 0; u < _vertexPosition.cols; u++) {
    for (int v = 0; v < _vertexPosition.rows; v++) {
      cv::Point3f vertex = _vertexPosition.at<cv::Point3f>(v, u);
      if (vertex.z > 0.0f &&  vertex.z < 1000.0f) {
        totalValidPixels++;
      }
    }
  }

  std::ofstream pointCloudFile;
  pointCloudFile.open (pointCloudPath);
  pointCloudFile << "ply\n";
  pointCloudFile << "format ascii 1.0\n";
  pointCloudFile << "element vertex " << totalValidPixels << "\n";
  pointCloudFile << "property float x\n";
  pointCloudFile << "property float y\n";
  pointCloudFile << "property float z\n";
  pointCloudFile << "property uchar red\n";
  pointCloudFile << "property uchar green\n";
  pointCloudFile << "property uchar blue\n";
  pointCloudFile << "end_header\n";

  for (int u = 0; u < _vertexPosition.cols; u++) {
    for (int v = 0; v < _vertexPosition.rows; v++) {
      cv::Point3f vertex = _vertexPosition.at<cv::Point3f>(v, u);
      cv::Vec3b color = _vertexColor.at<cv::Vec3b>(v, u);
      if (vertex.z > 0.0f &&  vertex.z < 1000.0f){
        pointCloudFile << vertex.x << " " << vertex.y << " " << vertex.z << " ";
        pointCloudFile << (int)color(0) << " " << (int)color(1) << " " << (int)color(2) << "\n";
      }
    }
  }

  pointCloudFile.close();
}
