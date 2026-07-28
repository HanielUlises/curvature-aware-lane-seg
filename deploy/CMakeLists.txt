cmake_minimum_required(VERSION 3.16)
project(curvature_port LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Eigen: prefer a config package, fall back to the common system include dir.
find_package(Eigen3 QUIET)
if(NOT TARGET Eigen3::Eigen)
  find_path(EIGEN3_INCLUDE_DIR Eigen/Dense
            PATHS /usr/include/eigen3 /usr/local/include/eigen3)
  if(NOT EIGEN3_INCLUDE_DIR)
    message(FATAL_ERROR "Eigen3 not found (apt install libeigen3-dev)")
  endif()
endif()

add_library(curvature_port src/curvature.cpp)
target_include_directories(curvature_port PUBLIC include)
if(TARGET Eigen3::Eigen)
  target_link_libraries(curvature_port PUBLIC Eigen3::Eigen)
else()
  target_include_directories(curvature_port PUBLIC ${EIGEN3_INCLUDE_DIR})
endif()
target_compile_options(curvature_port PRIVATE -Wall -Wextra)

enable_testing()
add_executable(test_curvature test/test_curvature.cpp)
target_link_libraries(test_curvature PRIVATE curvature_port)
target_compile_definitions(test_curvature PRIVATE
                           GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
add_test(NAME curvature_golden COMMAND test_curvature)
