cmake_minimum_required(VERSION 3.10)
project(mtmc_rknn VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Build natively on the OrangePi:
#   cmake -S . -B build -DRKNN_MODEL_ZOO=/path/to/rknn_model_zoo      (uses 3rdparty/rknpu2)
#   cmake -S . -B build -DRKNN_API_DIR=/path/to/dir                   (dir with include/rknn_api.h, lib/librknnrt.so)
#   cmake -S . -B build                                               (system rknn_api.h + librknnrt.so)
#   cmake --build build -j4
set(RKNN_MODEL_ZOO "" CACHE PATH "rknn_model_zoo checkout (optional)")
set(RKNN_API_DIR "" CACHE PATH "directory with include/rknn_api.h and lib/librknnrt.so (optional)")

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio highgui)
find_package(Threads REQUIRED)

if (RKNN_API_DIR)
    set(_inc ${RKNN_API_DIR}/include)
    find_library(RKNNRT_LIB rknnrt PATHS ${RKNN_API_DIR}/lib NO_DEFAULT_PATH)
elseif (RKNN_MODEL_ZOO)
    set(_inc ${RKNN_MODEL_ZOO}/3rdparty/rknpu2/include)
    find_library(RKNNRT_LIB rknnrt PATHS ${RKNN_MODEL_ZOO}/3rdparty/rknpu2/Linux/aarch64 NO_DEFAULT_PATH)
else()
    find_path(_inc rknn_api.h)
    find_library(RKNNRT_LIB rknnrt)
endif()
if (NOT _inc OR NOT RKNNRT_LIB)
    message(FATAL_ERROR "rknn_api.h / librknnrt.so not found - set -DRKNN_MODEL_ZOO or -DRKNN_API_DIR")
endif()
message(STATUS "RKNN include: ${_inc}")
message(STATUS "RKNN lib:     ${RKNNRT_LIB}")

add_executable(${PROJECT_NAME}
    main.cpp
    RknnModel.cpp
    Detector.cpp
    Geometry.cpp
    Tracker.cpp
    Renderer.cpp
)
target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS} ${_inc} ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS} ${RKNNRT_LIB} Threads::Threads)
