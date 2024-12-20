cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project(slow-node-detector VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)

find_package(MPI REQUIRED)
find_package(KokkosKernels REQUIRED)

add_subdirectory(src)

set(SLOW_NODE_EXE slow_node)

add_executable(
  ${SLOW_NODE_EXE}
  "${CMAKE_SOURCE_DIR}/src/slow_node.cc"
)

target_link_libraries(${SLOW_NODE_EXE} PUBLIC MPI::MPI_CXX)
target_link_libraries(${SLOW_NODE_EXE} PUBLIC Kokkos::kokkoskernels)
