cmake_minimum_required(VERSION 2.8.4)
project(Runge_Kutta)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp)
add_executable(Runge_Kutta ${SOURCE_FILES})