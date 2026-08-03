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
                           src/lane_tracker.cpp src/temporal.cpp src/mpc.cpp
                           src/preprocess.cpp)
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

add_executable(test_preprocess test/test_preprocess.cpp)
target_link_libraries(test_preprocess PRIVATE curvature_port)
target_compile_definitions(test_preprocess PRIVATE
                           GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
add_test(NAME preprocess_golden COMMAND test_preprocess)

add_executable(test_control test/test_control.cpp)
target_link_libraries(test_control PRIVATE curvature_port)
target_compile_definitions(test_control PRIVATE
                           GOLDEN_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test/golden")
add_test(NAME control_golden COMMAND test_control)

# Deployment showcase: the whole chain over a recorded sequence, with latencies.
add_executable(run_chain tools/run_chain.cpp)
target_link_libraries(run_chain PRIVATE curvature_port)

# ---------------------------------------------------------------------------
# The network stage, which is optional on purpose.
#
# curvature_port above is a few hundred kilobytes and needs nothing but Eigen; that is
# what makes it plausible on a vehicle ECU. ONNX Runtime and OpenCV are neither small
# nor always available, so the segmenter is a separate target and everything above it
# still builds when they are missing. A deployment that has its own inference runtime
# links curvature_port and calls the chain with a mask from wherever it likes.
#
# Point the build at an ONNX Runtime distribution with -DONNXRUNTIME_ROOT=<dir> (the
# unpacked release archive, containing include/ and lib/) or the environment variable
# of the same name.
set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}" CACHE PATH
    "Root of an ONNX Runtime distribution (include/ and lib/)")

find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
          HINTS ${ONNXRUNTIME_ROOT}/include
          PATHS /usr/local/include/onnxruntime /opt/onnxruntime/include)
find_library(ONNXRUNTIME_LIBRARY onnxruntime
             HINTS ${ONNXRUNTIME_ROOT}/lib
             PATHS /usr/local/lib /opt/onnxruntime/lib)
# Only looked for once ONNX Runtime is present, so that the common case — building the
# geometry chain alone — never touches OpenCV's config at all. That is not fastidiousness:
# an OpenCV built against a CUDA that is no longer installed aborts the configure step
# from inside its own config file, and a build that does not need OpenCV should not be
# able to fail that way. Set -DOpenCV_DIR=<dir> to choose between several installed.
if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIBRARY)
  find_package(OpenCV QUIET COMPONENTS core imgproc imgcodecs videoio)
endif()

if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIBRARY AND OpenCV_FOUND)
  message(STATUS "Segmenter enabled: ${ONNXRUNTIME_LIBRARY}")

  add_library(curvature_port_infer src/segmenter.cpp)
  target_include_directories(curvature_port_infer
                             PUBLIC include ${ONNXRUNTIME_INCLUDE_DIR})
  target_link_libraries(curvature_port_infer PUBLIC curvature_port
                        PRIVATE ${ONNXRUNTIME_LIBRARY} opencv_core opencv_imgproc)
  target_compile_options(curvature_port_infer PRIVATE -Wall -Wextra)
  set_property(TARGET curvature_port_infer PROPERTY POSITION_INDEPENDENT_CODE ON)
  # So the binaries find the runtime without the caller having to set LD_LIBRARY_PATH
  # for it. TensorRT's own libraries still need to be on the path; they are dlopened by
  # ONNX Runtime, not linked here.
  get_filename_component(_ort_libdir "${ONNXRUNTIME_LIBRARY}" DIRECTORY)
  set_property(TARGET curvature_port_infer APPEND PROPERTY
               INTERFACE_LINK_OPTIONS "-Wl,-rpath,${_ort_libdir}")

  # The whole pipeline over recorded footage: frames in, steering out, latencies split
  # three ways so the remaining budget is attributable.
  # Through the C ABI rather than the C++ classes, deliberately: it is the composite
  # entry point Python calls, so the showcase and the wrapper exercise the same code.
  add_executable(run_infer tools/run_infer.cpp)
  target_link_libraries(run_infer PRIVATE curvature_port_c curvature_port_infer
                        opencv_core opencv_imgproc opencv_imgcodecs opencv_videoio)

  # Backend comparison, which used to be a Python script and had no business being one:
  # it measures C++ runtimes and was reporting the timings of a Python process.
  add_executable(bench_backends tools/bench_backends.cpp)
  target_link_libraries(bench_backends PRIVATE curvature_port_infer
                        opencv_core opencv_imgproc opencv_imgcodecs)

  target_sources(curvature_port_c PRIVATE src/c_api_infer.cpp)
  target_link_libraries(curvature_port_c PRIVATE curvature_port_infer)
  target_compile_definitions(curvature_port_c PRIVATE CURVATURE_PORT_HAVE_SEGMENTER=1)
else()
  message(STATUS "Segmenter disabled (ONNX Runtime or OpenCV not found); the geometry "
                 "chain and its tests still build")
endif()
