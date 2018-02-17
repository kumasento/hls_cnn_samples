message(STATUS ${CMAKE_CURRENT_LIST_DIR})

file(GLOB SRC ${CMAKE_CURRENT_LIST_DIR}/layer.cc)

add_library(layer ${SRC})
target_include_directories(layer PUBLIC ${CMAKE_CURRENT_LIST_DIR})

add_executable(lenet ${CMAKE_CURRENT_LIST_DIR}/lenet.cc)
target_link_libraries(lenet layer)