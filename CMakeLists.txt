cmake_minimum_required(VERSION 3.9.1)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/;${CMAKE_MODULE_PATH}")
project(simple_petsc_example LANGUAGES CXX C)

add_subdirectory(src/c++/)
