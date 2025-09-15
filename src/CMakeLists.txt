set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

add_executable(create_index create_index.cpp)
target_link_libraries(create_index PRIVATE glog::glog fmt::fmt ${GFLAGS_LIBRARIES})

add_executable(test_qps test_qps.cpp)
target_link_libraries(test_qps PRIVATE glog::glog fmt::fmt ${GFLAGS_LIBRARIES})

# add_executable(test_qps_thread test_qps_thread.cpp)
# target_link_libraries(test_qps_thread PRIVATE glog::glog fmt::fmt ${GFLAGS_LIBRARIES})

add_executable(test_relative_error test_relative_error.cpp)
target_link_libraries(test_relative_error PRIVATE glog::glog fmt::fmt ${GFLAGS_LIBRARIES})

add_executable(compute_gt compute_gt.cpp)
target_link_libraries(compute_gt PRIVATE glog::glog fmt::fmt ${GFLAGS_LIBRARIES})

add_executable(test_ivf test_ivf.cpp)
target_link_libraries(test_ivf PRIVATE glog::glog fmt::fmt ${GFLAGS_LIBRARIES})
