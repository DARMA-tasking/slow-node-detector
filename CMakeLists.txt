cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project(slow-node-detector VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

add_subdirectory(src)

set(SLOW_NODE_EXE slow_node)

add_executable(
  ${SLOW_NODE_EXE}
  "${CMAKE_SOURCE_DIR}/src/slow_node.cc"
  "${CMAKE_SOURCE_DIR}/src/sensors.cc"
  "${CMAKE_SOURCE_DIR}/src/freq.cc"
)

find_package(MPI REQUIRED)
target_link_libraries(${SLOW_NODE_EXE} PUBLIC MPI::MPI_CXX)

find_package(Trilinos)
if (Trilinos_FOUND)
  find_package(MKL QUIET REQUIRED)
  target_link_libraries(${SLOW_NODE_EXE} PUBLIC MKL::MKL)
  target_link_libraries(${SLOW_NODE_EXE} PUBLIC ${Trilinos_LIBRARIES})
else()
  find_package(KokkosKernels REQUIRED)
  target_link_libraries(${SLOW_NODE_EXE} PUBLIC Kokkos::kokkoskernels)
endif()
