function(dh_add_coverage_test target_name source_file labels)
    add_executable(${target_name} ${source_file})
    target_link_libraries(${target_name} PRIVATE digital_human_core)
    target_include_directories(${target_name} PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src)
    set_target_properties(${target_name} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    add_test(NAME ${target_name} COMMAND ${target_name})
    set_tests_properties(${target_name} PROPERTIES
        LABELS "unit;coverage;${labels}"
        TIMEOUT 30)
endfunction()

dh_add_coverage_test(core_infrastructure_coverage_test
    core_infrastructure_coverage_test.cpp "core;observability")
dh_add_coverage_test(streaming_vad_coverage_test
    streaming_vad_coverage_test.cpp "audio")
dh_add_coverage_test(public_api_coverage_test
    public_api_coverage_test.cpp "api")
dh_add_coverage_test(module_contract_coverage_test
    module_contract_coverage_test.cpp "avatar;dialog;model")

set(DH_COVERAGE_TEST_TARGETS
    core_infrastructure_coverage_test
    streaming_vad_coverage_test
    public_api_coverage_test
    module_contract_coverage_test)

if(TARGET dh_network)
    add_executable(network_contract_coverage_test
        network_contract_coverage_test.cpp)
    target_include_directories(network_contract_coverage_test PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src)
    target_link_libraries(network_contract_coverage_test PRIVATE
        digital_human_core dh_network)
    set_target_properties(network_contract_coverage_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    add_test(NAME network_contract_coverage_test
        COMMAND network_contract_coverage_test)
    set_tests_properties(network_contract_coverage_test PROPERTIES
        LABELS "unit;coverage;network"
        TIMEOUT 30)
    list(APPEND DH_COVERAGE_TEST_TARGETS network_contract_coverage_test)
endif()

if(TARGET dh_media)
    add_executable(media_contract_coverage_test
        media_contract_coverage_test.cpp)
    target_include_directories(media_contract_coverage_test PRIVATE
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/src)
    target_link_libraries(media_contract_coverage_test PRIVATE
        digital_human_core dh_media)
    set_target_properties(media_contract_coverage_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
    add_test(NAME media_contract_coverage_test
        COMMAND media_contract_coverage_test)
    set_tests_properties(media_contract_coverage_test PROPERTIES
        LABELS "unit;coverage;media"
        TIMEOUT 30)
    list(APPEND DH_COVERAGE_TEST_TARGETS media_contract_coverage_test)
endif()

if(DIGITAL_HUMAN_ENABLE_COVERAGE)
    find_program(GCOVR_EXECUTABLE gcovr)
    if(GCOVR_EXECUTABLE)
        add_custom_target(coverage_report
            COMMAND ${CMAKE_COMMAND} -E make_directory
                ${CMAKE_BINARY_DIR}/coverage
            COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure -L coverage
            COMMAND ${GCOVR_EXECUTABLE}
                --root ${PROJECT_SOURCE_DIR}
                --filter ${PROJECT_SOURCE_DIR}/src
                --exclude ${PROJECT_SOURCE_DIR}/tests
                --html-details ${CMAKE_BINARY_DIR}/coverage/index.html
                --xml-pretty --xml ${CMAKE_BINARY_DIR}/coverage/coverage.xml
                --print-summary
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            DEPENDS ${DH_COVERAGE_TEST_TARGETS}
            COMMENT "Run deterministic coverage tests and generate HTML/XML reports")
    else()
        message(WARNING
            "Coverage instrumentation is enabled, but gcovr was not found; "
            "install gcovr to enable the coverage_report target.")
    endif()
endif()
