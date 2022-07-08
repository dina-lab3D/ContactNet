cmake_minimum_required(VERSION 3.8)
project(SelfDistograms)
set(CMAKE_CXX_STANDARD 14)

set(GAMB_LIB /cs/labs/dina/dina/libs/gamb++)
include_directories(${GAMB_LIB})
link_directories(${GAMB_LIB})

set(SOURCE_FILES cnpy.cpp distogram_maker.cpp)
add_executable(self_distogram_maker ${SOURCE_FILES})
target_link_libraries(self_distogram_maker gamb++ z)
