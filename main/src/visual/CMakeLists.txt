#add_library(sim_init sim_init.cpp)
#target_include_directories(sim_init PRIVATE ${PROJECT_SOURCE_DIR}/main/src ${COOLING_DIR} ${CSTONE_DIR} ${SPH_DIR}
#        ${MPI_CXX_INCLUDE_PATH})
#target_link_libraries(sim_init PRIVATE ${MPI_CXX_LIBRARIES} OpenMP::OpenMP_CXX)

set(VISUAL_GPU_KERNELS
        render_small.cu
)

if (CMAKE_HIP_COMPILER)
    set_source_files_properties(${VISUAL_GPU_KERNELS} PROPERTIES LANGUAGE HIP)
endif ()

if (CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER)
    add_library(visual_gpu_kernels OBJECT ${VISUAL_GPU_KERNELS})
    target_include_directories(visual_gpu_kernels PRIVATE ${PROJECT_SOURCE_DIR}/main/src ${COOLING_DIR} ${CSTONE_DIR}
            ${SPH_DIR} ${MPI_CXX_INCLUDE_PATH})
    target_compile_definitions(visual_gpu_kernels PRIVATE USE_CUDA)
    target_link_libraries(visual_gpu_kernels PRIVATE ${MPI_CXX_LIBRARIES} OpenMP::OpenMP_CXX cstone_gpu)
endif ()
#
#if (CMAKE_CUDA_COMPILER)
#    target_link_libraries(visual_gpu_kernels PRIVATE CUDA::cudart)
#endif ()

if (CMAKE_HIP_COMPILER)
    target_link_libraries(visual_gpu_kernels PRIVATE hip::device hip::host roc::rocthrust hip::hipcub)
    target_compile_definitions(visual_gpu_kernels PRIVATE THRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_HIP)
    #    set_target_properties(visual_gpu PROPERTIES LINKER_LANGUAGE CXX)
endif ()

if (CMAKE_CUDA_COMPILER OR CMAKE_HIP_COMPILER)
    add_library(visual_gpu
            $<TARGET_OBJECTS:visual_gpu_kernels>
    )
    set_target_properties(visual_gpu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif ()