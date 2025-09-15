cmake_minimum_required(VERSION 3.10)

project(SAQ LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(DEBUG_WITH_ASAN "Enable address sanitizer in debug compile mode" ON)
option(BUILD_UNIT_TESTS "Build unit tests with Google Test" ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

# AVX512 required
include(CheckCXXCompilerFlag)
include(CheckCXXSourceCompiles)

# Check if the compiler supports AVX-512 flags
check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512F)
check_cxx_compiler_flag("-mavx512dq" COMPILER_SUPPORTS_AVX512DQ)
check_cxx_compiler_flag("-mavx512bw" COMPILER_SUPPORTS_AVX512BW)
check_cxx_compiler_flag("-mavx512vl" COMPILER_SUPPORTS_AVX512VL)

# Test if AVX-512 can actually compile and run
set(CMAKE_REQUIRED_FLAGS "-mavx512f")
check_cxx_source_compiles(
  "#include <immintrin.h>
   int main() {
       __m512 a = _mm512_setzero_ps();
       __m512 b = _mm512_setzero_ps();
       __m512 c = _mm512_add_ps(a, b);
       return 0;
   }"
  HAS_AVX512F)

if(COMPILER_SUPPORTS_AVX512F AND HAS_AVX512F)
  message(STATUS "AVX-512F support detected and enabled")
  add_compile_options(-mavx512f)
  add_compile_options(-mavx512dq)
  add_compile_options(-mavx512bw)
  add_compile_options(-mavx512vl)
else()
  message(FATAL_ERROR "AVX-512 support requested but not available")
endif()

# find_package(OpenMP)
find_package(Glog)
find_package(Fmt)
find_package(GFlags)

# ==== dependencies ====
if(NOT Glog_FOUND)
  message(
    FATAL_ERROR
      "Glog not found. Install Glog from the command line using the command(s) -\
    sudo apt-get install libgoogle-glog-dev")
endif(NOT Glog_FOUND)

if(NOT GFlags_FOUND)
  message(
    FATAL_ERROR
      "GFlags not found. Install GFlags from the command line using the command(s) --\
    sudo apt-get install libgflags-dev")
endif(NOT GFlags_FOUND)

if(NOT Fmt_FOUND)
  message(
    FATAL_ERROR
      "Fmt not found. Install Fmt from the command line using the command(s) --\
    sudo apt-get install libfmt-dev")
endif(NOT Fmt_FOUND)

# Find Google Test only if unit tests are enabled
if(BUILD_UNIT_TESTS)
  enable_testing()
  find_package(GTest REQUIRED)
  if(NOT GTest_FOUND)
    message(
      FATAL_ERROR
        "Google Test not found. Install Google Test or set BUILD_UNIT_TESTS=OFF"
    )
  endif()
  message(STATUS "Building unit tests with Google Test ${GTEST_INCLUDE_DIRS}")
  include(GoogleTest)
else()
  message(STATUS "Unit tests disabled (BUILD_UNIT_TESTS=OFF)")
endif()

# ==== set compile options ====
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -Wall -Wextra -march=native -fpic -ftree-vectorize -fexceptions -ffp-contract=off -fno-finite-math-only"
)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wno-unused -Wcast-align")

if(DEBUG_WITH_ASAN)
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fsanitize=address")
endif()

include_directories(${PROJECT_SOURCE_DIR}/saqlib)

add_subdirectory(src)

# Build unit tests only if enabled
if(BUILD_UNIT_TESTS)
  add_subdirectory(unit_test)
endif()
