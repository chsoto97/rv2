# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.13

# Default target executed when no arguments are given to make.
default_target: all

.PHONY : default_target

# Allow only one "make -f Makefile2" at a time, but pass parallelism.
.NOTPARALLEL:


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
CMAKE_BINARY_DIR = "/home/chui/Documents/RobotVision/Assignment 1/robot-vision"

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/usr/bin/cmake -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache

.PHONY : rebuild_cache/fast

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake cache editor..."
	/usr/bin/cmake-gui -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache

.PHONY : edit_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start "/home/chui/Documents/RobotVision/Assignment 1/robot-vision/CMakeFiles" "/home/chui/Documents/RobotVision/Assignment 1/robot-vision/CMakeFiles/progress.marks"
	$(MAKE) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start "/home/chui/Documents/RobotVision/Assignment 1/robot-vision/CMakeFiles" 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean

.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named robot-vision

# Build rule for target.
robot-vision: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 robot-vision
.PHONY : robot-vision

# fast build rule for target.
robot-vision/fast:
	$(MAKE) -f CMakeFiles/robot-vision.dir/build.make CMakeFiles/robot-vision.dir/build
.PHONY : robot-vision/fast

src/Main.o: src/Main.cpp.o

.PHONY : src/Main.o

# target to build an object file
src/Main.cpp.o:
	$(MAKE) -f CMakeFiles/robot-vision.dir/build.make CMakeFiles/robot-vision.dir/src/Main.cpp.o
.PHONY : src/Main.cpp.o

src/Main.i: src/Main.cpp.i

.PHONY : src/Main.i

# target to preprocess a source file
src/Main.cpp.i:
	$(MAKE) -f CMakeFiles/robot-vision.dir/build.make CMakeFiles/robot-vision.dir/src/Main.cpp.i
.PHONY : src/Main.cpp.i

src/Main.s: src/Main.cpp.s

.PHONY : src/Main.s

# target to generate assembly for a file
src/Main.cpp.s:
	$(MAKE) -f CMakeFiles/robot-vision.dir/build.make CMakeFiles/robot-vision.dir/src/Main.cpp.s
.PHONY : src/Main.cpp.s

src/ass1/CameraCalib.o: src/ass1/CameraCalib.cpp.o

.PHONY : src/ass1/CameraCalib.o

# target to build an object file
src/ass1/CameraCalib.cpp.o:
	$(MAKE) -f CMakeFiles/robot-vision.dir/build.make CMakeFiles/robot-vision.dir/src/ass1/CameraCalib.cpp.o
.PHONY : src/ass1/CameraCalib.cpp.o

src/ass1/CameraCalib.i: src/ass1/CameraCalib.cpp.i

.PHONY : src/ass1/CameraCalib.i

# target to preprocess a source file
src/ass1/CameraCalib.cpp.i:
	$(MAKE) -f CMakeFiles/robot-vision.dir/build.make CMakeFiles/robot-vision.dir/src/ass1/CameraCalib.cpp.i
.PHONY : src/ass1/CameraCalib.cpp.i

src/ass1/CameraCalib.s: src/ass1/CameraCalib.cpp.s

.PHONY : src/ass1/CameraCalib.s

# target to generate assembly for a file
src/ass1/CameraCalib.cpp.s:
	$(MAKE) -f CMakeFiles/robot-vision.dir/build.make CMakeFiles/robot-vision.dir/src/ass1/CameraCalib.cpp.s
.PHONY : src/ass1/CameraCalib.cpp.s

src/ass1/ImageUtils.o: src/ass1/ImageUtils.cpp.o

.PHONY : src/ass1/ImageUtils.o

# target to build an object file
src/ass1/ImageUtils.cpp.o:
	$(MAKE) -f CMakeFiles/robot-vision.dir/build.make CMakeFiles/robot-vision.dir/src/ass1/ImageUtils.cpp.o
.PHONY : src/ass1/ImageUtils.cpp.o

src/ass1/ImageUtils.i: src/ass1/ImageUtils.cpp.i

.PHONY : src/ass1/ImageUtils.i

# target to preprocess a source file
src/ass1/ImageUtils.cpp.i:
	$(MAKE) -f CMakeFiles/robot-vision.dir/build.make CMakeFiles/robot-vision.dir/src/ass1/ImageUtils.cpp.i
.PHONY : src/ass1/ImageUtils.cpp.i

src/ass1/ImageUtils.s: src/ass1/ImageUtils.cpp.s

.PHONY : src/ass1/ImageUtils.s

# target to generate assembly for a file
src/ass1/ImageUtils.cpp.s:
	$(MAKE) -f CMakeFiles/robot-vision.dir/build.make CMakeFiles/robot-vision.dir/src/ass1/ImageUtils.cpp.s
.PHONY : src/ass1/ImageUtils.cpp.s

src/ass1/PLYFile.o: src/ass1/PLYFile.cpp.o

.PHONY : src/ass1/PLYFile.o

# target to build an object file
src/ass1/PLYFile.cpp.o:
	$(MAKE) -f CMakeFiles/robot-vision.dir/build.make CMakeFiles/robot-vision.dir/src/ass1/PLYFile.cpp.o
.PHONY : src/ass1/PLYFile.cpp.o

src/ass1/PLYFile.i: src/ass1/PLYFile.cpp.i

.PHONY : src/ass1/PLYFile.i

# target to preprocess a source file
src/ass1/PLYFile.cpp.i:
	$(MAKE) -f CMakeFiles/robot-vision.dir/build.make CMakeFiles/robot-vision.dir/src/ass1/PLYFile.cpp.i
.PHONY : src/ass1/PLYFile.cpp.i

src/ass1/PLYFile.s: src/ass1/PLYFile.cpp.s

.PHONY : src/ass1/PLYFile.s

# target to generate assembly for a file
src/ass1/PLYFile.cpp.s:
	$(MAKE) -f CMakeFiles/robot-vision.dir/build.make CMakeFiles/robot-vision.dir/src/ass1/PLYFile.cpp.s
.PHONY : src/ass1/PLYFile.cpp.s

src/ass1/StereoCalib.o: src/ass1/StereoCalib.cpp.o

.PHONY : src/ass1/StereoCalib.o

# target to build an object file
src/ass1/StereoCalib.cpp.o:
	$(MAKE) -f CMakeFiles/robot-vision.dir/build.make CMakeFiles/robot-vision.dir/src/ass1/StereoCalib.cpp.o
.PHONY : src/ass1/StereoCalib.cpp.o

src/ass1/StereoCalib.i: src/ass1/StereoCalib.cpp.i

.PHONY : src/ass1/StereoCalib.i

# target to preprocess a source file
src/ass1/StereoCalib.cpp.i:
	$(MAKE) -f CMakeFiles/robot-vision.dir/build.make CMakeFiles/robot-vision.dir/src/ass1/StereoCalib.cpp.i
.PHONY : src/ass1/StereoCalib.cpp.i

src/ass1/StereoCalib.s: src/ass1/StereoCalib.cpp.s

.PHONY : src/ass1/StereoCalib.s

# target to generate assembly for a file
src/ass1/StereoCalib.cpp.s:
	$(MAKE) -f CMakeFiles/robot-vision.dir/build.make CMakeFiles/robot-vision.dir/src/ass1/StereoCalib.cpp.s
.PHONY : src/ass1/StereoCalib.cpp.s

src/ass1/StereoMatch.o: src/ass1/StereoMatch.cpp.o

.PHONY : src/ass1/StereoMatch.o

# target to build an object file
src/ass1/StereoMatch.cpp.o:
	$(MAKE) -f CMakeFiles/robot-vision.dir/build.make CMakeFiles/robot-vision.dir/src/ass1/StereoMatch.cpp.o
.PHONY : src/ass1/StereoMatch.cpp.o

src/ass1/StereoMatch.i: src/ass1/StereoMatch.cpp.i

.PHONY : src/ass1/StereoMatch.i

# target to preprocess a source file
src/ass1/StereoMatch.cpp.i:
	$(MAKE) -f CMakeFiles/robot-vision.dir/build.make CMakeFiles/robot-vision.dir/src/ass1/StereoMatch.cpp.i
.PHONY : src/ass1/StereoMatch.cpp.i

src/ass1/StereoMatch.s: src/ass1/StereoMatch.cpp.s

.PHONY : src/ass1/StereoMatch.s

# target to generate assembly for a file
src/ass1/StereoMatch.cpp.s:
	$(MAKE) -f CMakeFiles/robot-vision.dir/build.make CMakeFiles/robot-vision.dir/src/ass1/StereoMatch.cpp.s
.PHONY : src/ass1/StereoMatch.cpp.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... rebuild_cache"
	@echo "... robot-vision"
	@echo "... edit_cache"
	@echo "... src/Main.o"
	@echo "... src/Main.i"
	@echo "... src/Main.s"
	@echo "... src/ass1/CameraCalib.o"
	@echo "... src/ass1/CameraCalib.i"
	@echo "... src/ass1/CameraCalib.s"
	@echo "... src/ass1/ImageUtils.o"
	@echo "... src/ass1/ImageUtils.i"
	@echo "... src/ass1/ImageUtils.s"
	@echo "... src/ass1/PLYFile.o"
	@echo "... src/ass1/PLYFile.i"
	@echo "... src/ass1/PLYFile.s"
	@echo "... src/ass1/StereoCalib.o"
	@echo "... src/ass1/StereoCalib.i"
	@echo "... src/ass1/StereoCalib.s"
	@echo "... src/ass1/StereoMatch.o"
	@echo "... src/ass1/StereoMatch.i"
	@echo "... src/ass1/StereoMatch.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -S$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system

