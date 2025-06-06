cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project(slow-node-detector VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

add_subdirectory(src)

set(SLOW_NODE_EXE slow_node)

add_executable(
  ${SLOW_NODE_EXE}
  "${CMAKE_SOURCE_DIR}/src/slow_node.cc"
  "${CMAKE_SOURCE_DIR}/src/benchmarks.cc"
  "${CMAKE_SOURCE_DIR}/src/sensors.cc"
  "${CMAKE_SOURCE_DIR}/src/freq.cc"
)

find_package(MPI REQUIRED)
target_link_libraries(${SLOW_NODE_EXE} PUBLIC MPI::MPI_CXX)

find_package(MKL QUIET REQUIRED)
target_link_libraries(${SLOW_NODE_EXE} PUBLIC MKL::MKL)

find_package(Trilinos QUIET)
if (Trilinos_FOUND)
  message(STATUS "Trilinos found: ${Trilinos_VERSION}")
  target_link_libraries(${SLOW_NODE_EXE} PUBLIC Kokkos::kokkos)
else()
  message(STATUS "Did not find Trilinos. Searching for kokkos directly.")
  find_package(Kokkos REQUIRED)
  target_link_libraries(${SLOW_NODE_EXE} PUBLIC Kokkos::kokkos)
endif()
