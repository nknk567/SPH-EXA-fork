include(cstone_add_test)

function(cstone_add_performance_test sourcename testname)
    add_executable(${testname} ${sourcename})
    target_include_directories(${testname} PRIVATE ${PROJECT_SOURCE_DIR}/include)
    target_include_directories(${testname} PRIVATE ${PROJECT_SOURCE_DIR}/test)
    target_link_libraries(${testname} PRIVATE OpenMP::OpenMP_CXX)
    install(TARGETS ${testname} RUNTIME DESTINATION ${CMAKE_INSTALL_SBINDIR}/performance)
endfunction()

function(cstone_add_cuda_performance_test sourcename testname)
    add_executable(${testname} ${objectname} ${sourcename})
    target_include_directories(${testname} PRIVATE ${PROJECT_SOURCE_DIR}/include)
    target_include_directories(${testname} PRIVATE ${PROJECT_SOURCE_DIR}/test)
    target_link_libraries(${testname} PRIVATE cstone_gpu ${CUDA_RUNTIME_LIBRARY} OpenMP::OpenMP_CXX)
    install(TARGETS ${testname} RUNTIME DESTINATION ${CMAKE_INSTALL_SBINDIR}/performance)
endfunction()
