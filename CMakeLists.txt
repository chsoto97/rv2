# Created by Braulio Sespede (15-3-2020)
cmake_minimum_required(VERSION 3.5)
project(robot-vision LANGUAGES CXX)
set(CMAKE_CXX_COMPILER g++)

# Find OpenCV library
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIB_DIR})

# Find Boost libraries
find_package(Boost REQUIRED filesystem)
link_directories(${Boost_INCLUDE_DIR})

# Compile C++ project
include_directories(include)
add_executable(${PROJECT_NAME}
		"src/Main.cpp"
		"src/ass1/ImageUtils.cpp" "include/ass1/ImageUtils.h"
		"src/ass1/CameraCalib.cpp" "include/ass1/CameraCalib.h"
		"src/ass1/StereoCalib.cpp" "include/ass1/StereoCalib.h"
		"src/ass1/StereoMatch.cpp" "include/ass1/StereoMatch.h"
		"src/ass1/PLYFile.cpp" "include/ass1/PLYFile.h")

# Link project with libraries
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${Boost_LIBRARIES})
