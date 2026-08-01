cmake_minimum_required(VERSION 3.16)
project(curvature_port LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# The per-frame path is the mask decomposition and the tracker, so the release build is
# tuned for it: -O3, and NDEBUG to drop Eigen's per-access assertions. Native tuning is
# opt-in rather than default, since a binary built with it will not run on an older CPU
# than the one that built it, which is the wrong default for a deployment artefact.
option(CURVATURE_PORT_NATIVE "Tune for the building machine's CPU" OFF)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
if(CURVATURE_PORT_NATIVE)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
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

add_library(curvature_port src/spline.cpp src/curvature.cpp src/ipm.cpp
                           src/road_geometry.cpp src/centerline.cpp
                           src/lane_tracker.cpp src/temporal.cpp src/mpc.cpp)
target_include_directories(curvature_port PUBLIC include PRIVATE src)
if(TARGET Eigen3::Eigen)
  target_link_libraries(curvature_port PUBLIC Eigen3::Eigen)
else()
  target_include_directories(curvature_port PUBLIC ${EIGEN3_INCLUDE_DIR})
endif()
target_compile_options(curvature_port PRIVATE -Wall -Wextra)
set_property(TARGET curvature_port PROPERTY POSITION_INDEPENDENT_CODE ON)

# Shared library with a C ABI, so Python drives the same code the vehicle runs instead of
# carrying a second implementation of every stage.
add_library(curvature_port_c SHARED src/c_api.cpp)
target_link_libraries(curvature_port_c PRIVATE curvature_port)
target_compile_options(curvature_port_c PRIVATE -Wall -Wextra)

enable_testing()
add_executable(test_curvature test/test_curvature.cpp)
target_link_libraries(test_curvature PRIVATE curvature_port)
target_compile_definitions(test_curvature PRIVATE
                           GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
add_test(NAME curvature_golden COMMAND test_curvature)

add_executable(test_geometry test/test_geometry.cpp)
target_link_libraries(test_geometry PRIVATE curvature_port)
target_compile_definitions(test_geometry PRIVATE
                           GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
add_test(NAME geometry_golden COMMAND test_geometry)

add_executable(test_pipeline test/test_pipeline.cpp)
target_link_libraries(test_pipeline PRIVATE curvature_port)
target_compile_definitions(test_pipeline PRIVATE
                           GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
add_test(NAME pipeline_golden COMMAND test_pipeline)

add_executable(test_control test/test_control.cpp)
target_link_libraries(test_control PRIVATE curvature_port)
target_compile_definitions(test_control PRIVATE
                           GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
add_test(NAME control_golden COMMAND test_control)

# Deployment showcase: the whole chain over a recorded sequence, with latencies.
add_executable(run_chain tools/run_chain.cpp)
target_link_libraries(run_chain PRIVATE curvature_port)
