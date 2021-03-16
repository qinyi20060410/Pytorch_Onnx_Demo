cmake_minimum_required(VERSION 3.0)

project(I)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
# set(CMAKE_BUILD_TYPE Debug)

# cuda
find_package(CUDA REQUIRED)
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

# tensorrt
include_directories(/home/terminal/Software/TensorRT-7.2.1.6/include)
link_directories(/home/terminal/Software/TensorRT-7.2.1.6/lib)

include_directories(/usr/include)
link_directories(/usr/lib)
include_directories(/usr/local/include)
link_directories(/usr/local/lib)
include_directories(${PROJECT_SOURCE_DIR}/common)
include_directories(${PROJECT_SOURCE_DIR}/include)

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})
link_libraries(${Opencv_LIBS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

# cuda_add_library(yololayer SHARED ${PROJECT_SOURCE_DIR}/yoloLayer.cu)
# target_link_libraries(yololayer nvinfer nvinferplugin cudart ${OpenCV_LIBS})


# add_executable(vpsIferFromOnnx
#     ${PROJECT_SOURCE_DIR}/src/infer.cpp
#     ${PROJECT_SOURCE_DIR}/common/getOptions.cpp
#     ${PROJECT_SOURCE_DIR}/common/logger.cpp
#     ${PROJECT_SOURCE_DIR}/common/TrtEngines.cpp
#     ${PROJECT_SOURCE_DIR}/common/TrtInference.cpp
#     ${PROJECT_SOURCE_DIR}/common/TrtOptions.cpp
#     ${PROJECT_SOURCE_DIR}/common/TrtReporting.cpp
# )
# target_link_libraries(vpsIferFromOnnx nvinfer)
# target_link_libraries(vpsIferFromOnnx cudart)
# target_link_libraries(vpsIferFromOnnx nvcaffe_parser)
# target_link_libraries(vpsIferFromOnnx nvinfer_plugin)
# target_link_libraries(vpsIferFromOnnx nvparsers)
# target_link_libraries(vpsIferFromOnnx nvonnxparser)
# target_link_libraries(vpsIferFromOnnx ${OpenCV_LIBS})
# target_link_libraries(vpsIferFromOnnx libyaml-cpp.so)
# target_link_libraries(vpsIferFromOnnx -ldl)
# target_link_libraries(vpsIferFromOnnx -pthread)


add_executable(inference
    ${PROJECT_SOURCE_DIR}/src/main.cpp
    ${PROJECT_SOURCE_DIR}/src/classify.cpp
    ${PROJECT_SOURCE_DIR}/common/logger.cpp
    ${PROJECT_SOURCE_DIR}/common/TrtEngines.cpp
    ${PROJECT_SOURCE_DIR}/common/TrtOptions.cpp
)
target_link_libraries(inference nvinfer)
target_link_libraries(inference cudart)
target_link_libraries(inference nvinfer_plugin)
target_link_libraries(inference nvparsers)
target_link_libraries(inference nvonnxparser)
target_link_libraries(inference ${OpenCV_LIBS})
target_link_libraries(inference libyaml-cpp.so)
target_link_libraries(inference -ldl)
target_link_libraries(inference -pthread)



add_definitions(-O2 -pthread)