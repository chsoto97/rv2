# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.13

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "/home/chui/Documents/RobotVision/Assignment 1/robot-vision"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/home/chui/Documents/RobotVision/Assignment 1/robot-vision/binaries"

# Include any dependencies generated for this target.
include CMakeFiles/robot-vision.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/robot-vision.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/robot-vision.dir/flags.make

CMakeFiles/robot-vision.dir/src/Main.cpp.o: CMakeFiles/robot-vision.dir/flags.make
CMakeFiles/robot-vision.dir/src/Main.cpp.o: ../src/Main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/home/chui/Documents/RobotVision/Assignment 1/robot-vision/binaries/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/robot-vision.dir/src/Main.cpp.o"
	g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/robot-vision.dir/src/Main.cpp.o -c "/home/chui/Documents/RobotVision/Assignment 1/robot-vision/src/Main.cpp"

CMakeFiles/robot-vision.dir/src/Main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/robot-vision.dir/src/Main.cpp.i"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/home/chui/Documents/RobotVision/Assignment 1/robot-vision/src/Main.cpp" > CMakeFiles/robot-vision.dir/src/Main.cpp.i

CMakeFiles/robot-vision.dir/src/Main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/robot-vision.dir/src/Main.cpp.s"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/home/chui/Documents/RobotVision/Assignment 1/robot-vision/src/Main.cpp" -o CMakeFiles/robot-vision.dir/src/Main.cpp.s

CMakeFiles/robot-vision.dir/src/ass1/ImageUtils.cpp.o: CMakeFiles/robot-vision.dir/flags.make
CMakeFiles/robot-vision.dir/src/ass1/ImageUtils.cpp.o: ../src/ass1/ImageUtils.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/home/chui/Documents/RobotVision/Assignment 1/robot-vision/binaries/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/robot-vision.dir/src/ass1/ImageUtils.cpp.o"
	g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/robot-vision.dir/src/ass1/ImageUtils.cpp.o -c "/home/chui/Documents/RobotVision/Assignment 1/robot-vision/src/ass1/ImageUtils.cpp"

CMakeFiles/robot-vision.dir/src/ass1/ImageUtils.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/robot-vision.dir/src/ass1/ImageUtils.cpp.i"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/home/chui/Documents/RobotVision/Assignment 1/robot-vision/src/ass1/ImageUtils.cpp" > CMakeFiles/robot-vision.dir/src/ass1/ImageUtils.cpp.i

CMakeFiles/robot-vision.dir/src/ass1/ImageUtils.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/robot-vision.dir/src/ass1/ImageUtils.cpp.s"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/home/chui/Documents/RobotVision/Assignment 1/robot-vision/src/ass1/ImageUtils.cpp" -o CMakeFiles/robot-vision.dir/src/ass1/ImageUtils.cpp.s

CMakeFiles/robot-vision.dir/src/ass1/CameraCalib.cpp.o: CMakeFiles/robot-vision.dir/flags.make
CMakeFiles/robot-vision.dir/src/ass1/CameraCalib.cpp.o: ../src/ass1/CameraCalib.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/home/chui/Documents/RobotVision/Assignment 1/robot-vision/binaries/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/robot-vision.dir/src/ass1/CameraCalib.cpp.o"
	g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/robot-vision.dir/src/ass1/CameraCalib.cpp.o -c "/home/chui/Documents/RobotVision/Assignment 1/robot-vision/src/ass1/CameraCalib.cpp"

CMakeFiles/robot-vision.dir/src/ass1/CameraCalib.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/robot-vision.dir/src/ass1/CameraCalib.cpp.i"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/home/chui/Documents/RobotVision/Assignment 1/robot-vision/src/ass1/CameraCalib.cpp" > CMakeFiles/robot-vision.dir/src/ass1/CameraCalib.cpp.i

CMakeFiles/robot-vision.dir/src/ass1/CameraCalib.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/robot-vision.dir/src/ass1/CameraCalib.cpp.s"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/home/chui/Documents/RobotVision/Assignment 1/robot-vision/src/ass1/CameraCalib.cpp" -o CMakeFiles/robot-vision.dir/src/ass1/CameraCalib.cpp.s

CMakeFiles/robot-vision.dir/src/ass1/StereoCalib.cpp.o: CMakeFiles/robot-vision.dir/flags.make
CMakeFiles/robot-vision.dir/src/ass1/StereoCalib.cpp.o: ../src/ass1/StereoCalib.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/home/chui/Documents/RobotVision/Assignment 1/robot-vision/binaries/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/robot-vision.dir/src/ass1/StereoCalib.cpp.o"
	g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/robot-vision.dir/src/ass1/StereoCalib.cpp.o -c "/home/chui/Documents/RobotVision/Assignment 1/robot-vision/src/ass1/StereoCalib.cpp"

CMakeFiles/robot-vision.dir/src/ass1/StereoCalib.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/robot-vision.dir/src/ass1/StereoCalib.cpp.i"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/home/chui/Documents/RobotVision/Assignment 1/robot-vision/src/ass1/StereoCalib.cpp" > CMakeFiles/robot-vision.dir/src/ass1/StereoCalib.cpp.i

CMakeFiles/robot-vision.dir/src/ass1/StereoCalib.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/robot-vision.dir/src/ass1/StereoCalib.cpp.s"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/home/chui/Documents/RobotVision/Assignment 1/robot-vision/src/ass1/StereoCalib.cpp" -o CMakeFiles/robot-vision.dir/src/ass1/StereoCalib.cpp.s

CMakeFiles/robot-vision.dir/src/ass1/StereoMatch.cpp.o: CMakeFiles/robot-vision.dir/flags.make
CMakeFiles/robot-vision.dir/src/ass1/StereoMatch.cpp.o: ../src/ass1/StereoMatch.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/home/chui/Documents/RobotVision/Assignment 1/robot-vision/binaries/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/robot-vision.dir/src/ass1/StereoMatch.cpp.o"
	g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/robot-vision.dir/src/ass1/StereoMatch.cpp.o -c "/home/chui/Documents/RobotVision/Assignment 1/robot-vision/src/ass1/StereoMatch.cpp"

CMakeFiles/robot-vision.dir/src/ass1/StereoMatch.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/robot-vision.dir/src/ass1/StereoMatch.cpp.i"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/home/chui/Documents/RobotVision/Assignment 1/robot-vision/src/ass1/StereoMatch.cpp" > CMakeFiles/robot-vision.dir/src/ass1/StereoMatch.cpp.i

CMakeFiles/robot-vision.dir/src/ass1/StereoMatch.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/robot-vision.dir/src/ass1/StereoMatch.cpp.s"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/home/chui/Documents/RobotVision/Assignment 1/robot-vision/src/ass1/StereoMatch.cpp" -o CMakeFiles/robot-vision.dir/src/ass1/StereoMatch.cpp.s

CMakeFiles/robot-vision.dir/src/ass1/PLYFile.cpp.o: CMakeFiles/robot-vision.dir/flags.make
CMakeFiles/robot-vision.dir/src/ass1/PLYFile.cpp.o: ../src/ass1/PLYFile.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/home/chui/Documents/RobotVision/Assignment 1/robot-vision/binaries/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_6) "Building CXX object CMakeFiles/robot-vision.dir/src/ass1/PLYFile.cpp.o"
	g++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/robot-vision.dir/src/ass1/PLYFile.cpp.o -c "/home/chui/Documents/RobotVision/Assignment 1/robot-vision/src/ass1/PLYFile.cpp"

CMakeFiles/robot-vision.dir/src/ass1/PLYFile.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/robot-vision.dir/src/ass1/PLYFile.cpp.i"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/home/chui/Documents/RobotVision/Assignment 1/robot-vision/src/ass1/PLYFile.cpp" > CMakeFiles/robot-vision.dir/src/ass1/PLYFile.cpp.i

CMakeFiles/robot-vision.dir/src/ass1/PLYFile.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/robot-vision.dir/src/ass1/PLYFile.cpp.s"
	g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/home/chui/Documents/RobotVision/Assignment 1/robot-vision/src/ass1/PLYFile.cpp" -o CMakeFiles/robot-vision.dir/src/ass1/PLYFile.cpp.s

# Object files for target robot-vision
robot__vision_OBJECTS = \
"CMakeFiles/robot-vision.dir/src/Main.cpp.o" \
"CMakeFiles/robot-vision.dir/src/ass1/ImageUtils.cpp.o" \
"CMakeFiles/robot-vision.dir/src/ass1/CameraCalib.cpp.o" \
"CMakeFiles/robot-vision.dir/src/ass1/StereoCalib.cpp.o" \
"CMakeFiles/robot-vision.dir/src/ass1/StereoMatch.cpp.o" \
"CMakeFiles/robot-vision.dir/src/ass1/PLYFile.cpp.o"

# External object files for target robot-vision
robot__vision_EXTERNAL_OBJECTS =

robot-vision: CMakeFiles/robot-vision.dir/src/Main.cpp.o
robot-vision: CMakeFiles/robot-vision.dir/src/ass1/ImageUtils.cpp.o
robot-vision: CMakeFiles/robot-vision.dir/src/ass1/CameraCalib.cpp.o
robot-vision: CMakeFiles/robot-vision.dir/src/ass1/StereoCalib.cpp.o
robot-vision: CMakeFiles/robot-vision.dir/src/ass1/StereoMatch.cpp.o
robot-vision: CMakeFiles/robot-vision.dir/src/ass1/PLYFile.cpp.o
robot-vision: CMakeFiles/robot-vision.dir/build.make
robot-vision: /usr/lib/x86_64-linux-gnu/libopencv_shape.so.3.2.0
robot-vision: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.3.2.0
robot-vision: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.3.2.0
robot-vision: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.3.2.0
robot-vision: /usr/lib/x86_64-linux-gnu/libopencv_aruco.so.3.2.0
robot-vision: /usr/lib/x86_64-linux-gnu/libopencv_bgsegm.so.3.2.0
robot-vision: /usr/lib/x86_64-linux-gnu/libopencv_bioinspired.so.3.2.0
robot-vision: /usr/lib/x86_64-linux-gnu/libopencv_ccalib.so.3.2.0
robot-vision: /usr/lib/x86_64-linux-gnu/libopencv_datasets.so.3.2.0
robot-vision: /usr/lib/x86_64-linux-gnu/libopencv_dpm.so.3.2.0
robot-vision: /usr/lib/x86_64-linux-gnu/libopencv_face.so.3.2.0
robot-vision: /usr/lib/x86_64-linux-gnu/libopencv_freetype.so.3.2.0
robot-vision: /usr/lib/x86_64-linux-gnu/libopencv_fuzzy.so.3.2.0
robot-vision: /usr/lib/x86_64-linux-gnu/libopencv_hdf.so.3.2.0
robot-vision: /usr/lib/x86_64-linux-gnu/libopencv_line_descriptor.so.3.2.0
robot-vision: /usr/lib/x86_64-linux-gnu/libopencv_optflow.so.3.2.0
robot-vision: /usr/lib/x86_64-linux-gnu/libopencv_plot.so.3.2.0
robot-vision: /usr/lib/x86_64-linux-gnu/libopencv_reg.so.3.2.0
robot-vision: /usr/lib/x86_64-linux-gnu/libopencv_saliency.so.3.2.0
robot-vision: /usr/lib/x86_64-linux-gnu/libopencv_stereo.so.3.2.0
robot-vision: /usr/lib/x86_64-linux-gnu/libopencv_structured_light.so.3.2.0
robot-vision: /usr/lib/x86_64-linux-gnu/libopencv_surface_matching.so.3.2.0
robot-vision: /usr/lib/x86_64-linux-gnu/libopencv_text.so.3.2.0
robot-vision: /usr/lib/x86_64-linux-gnu/libopencv_ximgproc.so.3.2.0
robot-vision: /usr/lib/x86_64-linux-gnu/libopencv_xobjdetect.so.3.2.0
robot-vision: /usr/lib/x86_64-linux-gnu/libopencv_xphoto.so.3.2.0
robot-vision: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
robot-vision: /usr/lib/x86_64-linux-gnu/libboost_system.so
robot-vision: /usr/lib/x86_64-linux-gnu/libopencv_video.so.3.2.0
robot-vision: /usr/lib/x86_64-linux-gnu/libopencv_viz.so.3.2.0
robot-vision: /usr/lib/x86_64-linux-gnu/libopencv_phase_unwrapping.so.3.2.0
robot-vision: /usr/lib/x86_64-linux-gnu/libopencv_rgbd.so.3.2.0
robot-vision: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.3.2.0
robot-vision: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.3.2.0
robot-vision: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.3.2.0
robot-vision: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.3.2.0
robot-vision: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.3.2.0
robot-vision: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.3.2.0
robot-vision: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.3.2.0
robot-vision: /usr/lib/x86_64-linux-gnu/libopencv_videoio.so.3.2.0
robot-vision: /usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so.3.2.0
robot-vision: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.3.2.0
robot-vision: /usr/lib/x86_64-linux-gnu/libopencv_core.so.3.2.0
robot-vision: CMakeFiles/robot-vision.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir="/home/chui/Documents/RobotVision/Assignment 1/robot-vision/binaries/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_7) "Linking CXX executable robot-vision"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/robot-vision.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/robot-vision.dir/build: robot-vision

.PHONY : CMakeFiles/robot-vision.dir/build

CMakeFiles/robot-vision.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/robot-vision.dir/cmake_clean.cmake
.PHONY : CMakeFiles/robot-vision.dir/clean

CMakeFiles/robot-vision.dir/depend:
	cd "/home/chui/Documents/RobotVision/Assignment 1/robot-vision/binaries" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/home/chui/Documents/RobotVision/Assignment 1/robot-vision" "/home/chui/Documents/RobotVision/Assignment 1/robot-vision" "/home/chui/Documents/RobotVision/Assignment 1/robot-vision/binaries" "/home/chui/Documents/RobotVision/Assignment 1/robot-vision/binaries" "/home/chui/Documents/RobotVision/Assignment 1/robot-vision/binaries/CMakeFiles/robot-vision.dir/DependInfo.cmake" --color=$(COLOR)
.PHONY : CMakeFiles/robot-vision.dir/depend

