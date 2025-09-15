# Build unit tests only if enabled

add_executable(unit_tests ut_main.cpp ut_ivf_error.cpp
                          ut_ivf_recall.cpp ut_cluster_estimator.cpp
                          ut_single_estimator.cpp)
target_link_libraries(
  unit_tests PRIVATE glog::glog fmt::fmt ${GFLAGS_LIBRARIES} GTest::gtest
                     GTest::gtest_main)

set(CTEST_CUSTOM_MAXIMUM_PASSED_TEST_OUTPUT_SIZE 4096)
gtest_discover_tests(unit_tests WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
