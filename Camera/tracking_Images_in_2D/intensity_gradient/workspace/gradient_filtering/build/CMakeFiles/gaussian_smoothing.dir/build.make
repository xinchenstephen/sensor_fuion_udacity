# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.10

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
CMAKE_SOURCE_DIR = /mnt/d/c++vs/sensor_fusion/Camera/intensity_gradient/workspace/gradient_filtering

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /mnt/d/c++vs/sensor_fusion/Camera/intensity_gradient/workspace/gradient_filtering/build

# Include any dependencies generated for this target.
include CMakeFiles/gaussian_smoothing.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/gaussian_smoothing.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/gaussian_smoothing.dir/flags.make

CMakeFiles/gaussian_smoothing.dir/src/gaussian_smoothing.cpp.o: CMakeFiles/gaussian_smoothing.dir/flags.make
CMakeFiles/gaussian_smoothing.dir/src/gaussian_smoothing.cpp.o: ../src/gaussian_smoothing.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/d/c++vs/sensor_fusion/Camera/intensity_gradient/workspace/gradient_filtering/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/gaussian_smoothing.dir/src/gaussian_smoothing.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/gaussian_smoothing.dir/src/gaussian_smoothing.cpp.o -c /mnt/d/c++vs/sensor_fusion/Camera/intensity_gradient/workspace/gradient_filtering/src/gaussian_smoothing.cpp

CMakeFiles/gaussian_smoothing.dir/src/gaussian_smoothing.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/gaussian_smoothing.dir/src/gaussian_smoothing.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/d/c++vs/sensor_fusion/Camera/intensity_gradient/workspace/gradient_filtering/src/gaussian_smoothing.cpp > CMakeFiles/gaussian_smoothing.dir/src/gaussian_smoothing.cpp.i

CMakeFiles/gaussian_smoothing.dir/src/gaussian_smoothing.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/gaussian_smoothing.dir/src/gaussian_smoothing.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/d/c++vs/sensor_fusion/Camera/intensity_gradient/workspace/gradient_filtering/src/gaussian_smoothing.cpp -o CMakeFiles/gaussian_smoothing.dir/src/gaussian_smoothing.cpp.s

CMakeFiles/gaussian_smoothing.dir/src/gaussian_smoothing.cpp.o.requires:

.PHONY : CMakeFiles/gaussian_smoothing.dir/src/gaussian_smoothing.cpp.o.requires

CMakeFiles/gaussian_smoothing.dir/src/gaussian_smoothing.cpp.o.provides: CMakeFiles/gaussian_smoothing.dir/src/gaussian_smoothing.cpp.o.requires
	$(MAKE) -f CMakeFiles/gaussian_smoothing.dir/build.make CMakeFiles/gaussian_smoothing.dir/src/gaussian_smoothing.cpp.o.provides.build
.PHONY : CMakeFiles/gaussian_smoothing.dir/src/gaussian_smoothing.cpp.o.provides

CMakeFiles/gaussian_smoothing.dir/src/gaussian_smoothing.cpp.o.provides.build: CMakeFiles/gaussian_smoothing.dir/src/gaussian_smoothing.cpp.o


# Object files for target gaussian_smoothing
gaussian_smoothing_OBJECTS = \
"CMakeFiles/gaussian_smoothing.dir/src/gaussian_smoothing.cpp.o"

# External object files for target gaussian_smoothing
gaussian_smoothing_EXTERNAL_OBJECTS =

gaussian_smoothing: CMakeFiles/gaussian_smoothing.dir/src/gaussian_smoothing.cpp.o
gaussian_smoothing: CMakeFiles/gaussian_smoothing.dir/build.make
gaussian_smoothing: /usr/local/lib/libopencv_dnn.so.4.2.0
gaussian_smoothing: /usr/local/lib/libopencv_gapi.so.4.2.0
gaussian_smoothing: /usr/local/lib/libopencv_highgui.so.4.2.0
gaussian_smoothing: /usr/local/lib/libopencv_ml.so.4.2.0
gaussian_smoothing: /usr/local/lib/libopencv_objdetect.so.4.2.0
gaussian_smoothing: /usr/local/lib/libopencv_photo.so.4.2.0
gaussian_smoothing: /usr/local/lib/libopencv_stitching.so.4.2.0
gaussian_smoothing: /usr/local/lib/libopencv_video.so.4.2.0
gaussian_smoothing: /usr/local/lib/libopencv_videoio.so.4.2.0
gaussian_smoothing: /usr/local/lib/libopencv_imgcodecs.so.4.2.0
gaussian_smoothing: /usr/local/lib/libopencv_calib3d.so.4.2.0
gaussian_smoothing: /usr/local/lib/libopencv_features2d.so.4.2.0
gaussian_smoothing: /usr/local/lib/libopencv_flann.so.4.2.0
gaussian_smoothing: /usr/local/lib/libopencv_imgproc.so.4.2.0
gaussian_smoothing: /usr/local/lib/libopencv_core.so.4.2.0
gaussian_smoothing: CMakeFiles/gaussian_smoothing.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/mnt/d/c++vs/sensor_fusion/Camera/intensity_gradient/workspace/gradient_filtering/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable gaussian_smoothing"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/gaussian_smoothing.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/gaussian_smoothing.dir/build: gaussian_smoothing

.PHONY : CMakeFiles/gaussian_smoothing.dir/build

CMakeFiles/gaussian_smoothing.dir/requires: CMakeFiles/gaussian_smoothing.dir/src/gaussian_smoothing.cpp.o.requires

.PHONY : CMakeFiles/gaussian_smoothing.dir/requires

CMakeFiles/gaussian_smoothing.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/gaussian_smoothing.dir/cmake_clean.cmake
.PHONY : CMakeFiles/gaussian_smoothing.dir/clean

CMakeFiles/gaussian_smoothing.dir/depend:
	cd /mnt/d/c++vs/sensor_fusion/Camera/intensity_gradient/workspace/gradient_filtering/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/d/c++vs/sensor_fusion/Camera/intensity_gradient/workspace/gradient_filtering /mnt/d/c++vs/sensor_fusion/Camera/intensity_gradient/workspace/gradient_filtering /mnt/d/c++vs/sensor_fusion/Camera/intensity_gradient/workspace/gradient_filtering/build /mnt/d/c++vs/sensor_fusion/Camera/intensity_gradient/workspace/gradient_filtering/build /mnt/d/c++vs/sensor_fusion/Camera/intensity_gradient/workspace/gradient_filtering/build/CMakeFiles/gaussian_smoothing.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/gaussian_smoothing.dir/depend

