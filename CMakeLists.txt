cmake_minimum_required(VERSION 3.24)

project(greekstream
  VERSION 0.1.0
  DESCRIPTION "Real-time options risk engine in C++20 with expression-template adjoint algorithmic differentiation"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(QC_BUILD_TESTS "Build unit and property tests" ON)
option(QC_BUILD_BENCH "Build google/benchmark suites" ON)
option(QC_BUILD_TOOLS "Build the live risk demo" ON)
option(QC_WARNINGS_AS_ERRORS "Treat warnings as errors" ON)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(ProjectOptions)

find_package(Threads REQUIRED)

add_library(greekstream INTERFACE)
add_library(greekstream::greekstream ALIAS greekstream)
target_include_directories(greekstream INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(greekstream INTERFACE cxx_std_20)
target_link_libraries(greekstream INTERFACE Threads::Threads)

include(Dependencies)

if(QC_BUILD_TOOLS)
  add_executable(greekstream-demo tools/greekstream_demo.cpp)
  target_link_libraries(greekstream-demo PRIVATE greekstream::greekstream)
  qc_set_warnings(greekstream-demo)
endif()

if(QC_BUILD_TESTS)
  enable_testing()
  include(GoogleTest)
  add_executable(greekstream_tests
    tests/test_aad.cpp
    tests/test_models.cpp
    tests/test_monte_carlo.cpp
    tests/test_risk_engine.cpp)
  target_link_libraries(greekstream_tests PRIVATE greekstream::greekstream GTest::gtest_main)
  qc_set_warnings(greekstream_tests)
  gtest_discover_tests(greekstream_tests DISCOVERY_TIMEOUT 60 PROPERTIES TIMEOUT 300)
endif()

if(QC_BUILD_BENCH)
  add_executable(greekstream_bench bench/bench_greeks.cpp)
  target_link_libraries(greekstream_bench PRIVATE greekstream::greekstream benchmark::benchmark_main)
  qc_set_warnings(greekstream_bench)
endif()
