cmake_minimum_required(VERSION 3.19.0)
project(mx_driving)

set(CMAKE_SKIP_RPATH TRUE)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(cmake/config.cmake)
include(cmake/func.cmake)
include(cmake/intf.cmake)

option(USE_ARCH35 "Use arch35 specific implementations" OFF)

# 添加算子目录
add_subdirectory(${PROJECT_SOURCE_DIR}/kernels)

# 添加其他子目录
add_subdirectory(${PROJECT_SOURCE_DIR}/onnx_plugin)
add_subdirectory(${PROJECT_SOURCE_DIR}/mx_driving/csrc)
