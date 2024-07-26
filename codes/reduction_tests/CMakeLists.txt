cmake_minimum_required(VERSION 3.24)

include(CMakeDependentOption)

set(VERSION_MAJOR 0)
# anything above 12 is to indicate that it is devel-branch. The git hash commit
# is the only important information to give. It is retrieved few lines below.
set(VERSION_MINOR 0)
set(VERSION_PATCH 1)

project(
  reduce_tests
  DESCRIPTION "testing determinism of the reduce kernel"
  HOMEPAGE_URL "none"
  VERSION ${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}
  LANGUAGES C CXX)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# set language and standard
set(CMAKE_CXX_STANDARD 17)
option(REDUCE_USE_SCOREP "Use scorep to monitor power comsumption. Will also be turned on for gpu" OFF)
option(REDUCE_USE_CUDA "Compile with CUDA support" OFF)
option(REDUCE_USE_HIP "Compile with HIP support" OFF)

if ((REDUCE_USE_CUDA AND REDUCE_USE_HIP) OR ((NOT REDUCE_USE_CUDA) AND (NOT REDUCE_USE_HIP)))
  message(FATAL_ERROR "HIP and CUDA backend can not be both selected. Use either -DREDUCE_USE_CUDA or -DREDUCE_USE_HIP")
endif()

if (REDUCE_USE_SCOREP)
  set(CMAKE_CXX_COMPILER scorep-g++)
endif()

find_package(OpenMP REQUIRED COMPONENTS C CXX)

if (REDUCE_USE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  set(WHIP_BACKEND "CUDA")
  if (REDUCE_USE_SCOREP)
    set(CMAKE_CUDA_COMPILER "scopep --nocompiler ${CMAKE_CUDA_COMPILER}")
  endif()
endif ()

if (REDUCE_USE_HIP)
  enable_language(HIP)
  find_package(hiprand REQUIRED IMPORTED CONFIG)
  set(WHIP_BACKEND "HIP")
  if (REDUCE_USE_SCOREP)
    set(CMAKE_HIP_COMPILER "scopep --nocompiler ${CMAKE_HIP_COMPILER}")
  endif()
endif()
include(FetchContent)

##############################################################
# FetchContent_Declare(                                      #
#   rapidcsv                                                 #
#   GIT_REPOSITORY "https://github.com/d99kris/rapidcsv.git" #
#   GIT_TAG        "v8.50"                                   #
#   FIND_PACKAGE_ARGS NAMES rapidcsv                         #
# )                                                          #
# FetchContent_MakeAvailable(rapidcsv)                       #
##############################################################

FetchContent_Declare(
  cxxopts
  GIT_REPOSITORY "https://github.com/jarro2783/cxxopts.git"
  GIT_TAG        "v3.2.1"
  FIND_PACKAGE_ARGS NAMES cxxopts
)
FetchContent_MakeAvailable(cxxopts)

FetchContent_Declare(
  indicators
  GIT_REPOSITORY "https://github.com/p-ranav/indicators.git"
  GIT_TAG        "v2.3"
  FIND_PACKAGE_ARGS NAMES indicators
)
FetchContent_MakeAvailable(indicators)

FetchContent_Declare(
  fmt
  GIT_REPOSITORY "https://github.com/fmtlib/fmt"
  GIT_TAG        "10.2.1"
  FIND_PACKAGE_ARGS NAMES fmt
)
FetchContent_MakeAvailable(fmt)

###########################################################
# FetchContent_Declare(                                   #
#   whip                                                  #
#   GIT_REPOSITORY "https://github.com/eth-cscs/whip.git" #
#   GIT_TAG        "0.3.0"                                #
#   FIND_PACKAGE_ARGS NAMES whip                          #
# )                                                       #
# FetchContent_MakeAvailable(whip)                        #
###########################################################

add_executable(test_reduce ornl_reduce.cu reduce_test.cpp utils.cu timing/rt_graph.cpp)

if (REDUCE_USE_HIP)
	set_source_files_properties(ornl_reduce.cu utils.cu PROPERTIES LANGUAGE HIP)
endif()

foreach(_app test_reduce)
  set_target_properties(${_app} PROPERTIES IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
    LINKER_LANGUAGE "CXX")
  target_compile_features(${_app} PUBLIC cxx_std_17)
  target_compile_definitions(${_app} PRIVATE
	  $<$<BOOL:${REDUCE_USE_CUDA}>:REDUCE_USE_CUDA WHIP_CUDA>
	  $<$<BOOL:${REDUCE_USE_HIP}>:REDUCE_USE_HIP WHIP_HIP __HIP_PLATFORM_AMD__>)
  target_include_directories(${_app} PUBLIC ${CMAKE_SOURCE_DIR}/timing
    $<$<BOOL:${REDUCE_USE_HIP}>:${CMAKE_SOURCE_DIR}/whip/hip>
    $<$<BOOL:${REDUCE_USE_CUDA}>:${CMAKE_SOURCE_DIR}/whip/cuda>)
  target_link_libraries(${_app} PUBLIC
    $<$<BOOL:${REDUCE_USE_CUDA}>:CUDA::curand CUDA::cudart CUDA::cuda_driver>
    $<$<BOOL:${REDUCE_USE_HIP}>:hip::hiprand hip::host>
    cxxopts::cxxopts
    indicators::indicators
    fmt::fmt)
endforeach()
