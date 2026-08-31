file(GLOB_RECURSE FIXED_DIM_QP_TEST_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")

add_executable(fixed_dim_qp_tests ${FIXED_DIM_QP_TEST_SOURCES})
target_link_libraries(fixed_dim_qp_tests PRIVATE fixed_dim_qp Catch2::Catch2WithMain)
target_include_directories(fixed_dim_qp_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

include(CTest)
include(Catch)
catch_discover_tests(fixed_dim_qp_tests)
