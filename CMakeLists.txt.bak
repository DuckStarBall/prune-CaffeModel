cmake_minimum_required(VERSION 2.8)
project(PruneCaffemodel)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3")

set(CMAKE_PREFIX_PATH ${PROJECT_SOURCE_DIR}/../3rdparty-ubuntu16/caffe-cpu)
find_package(Caffe REQUIRED)
add_definitions(-DCPU_ONLY)

include_directories(${Caffe_INCLUDE_DIRS})
add_executable(prune_by_l1norm prune_by_l1norm.cpp)
target_link_libraries(prune_by_l1norm
        ${Caffe_LIBRARIES}
        )