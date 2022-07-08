cmake_minimum_required(VERSION 3.8)
project(InterfaceFeatures)
include(GNUInstallDirs)
set(CMAKE_CXX_STANDARD 17)

#FIND_PACKAGE(Boost COMPONENTS program_options REQUIRED)

set(GAMB_LIB /cs/labs/dina/dina/libs/gamb++)

include_directories(${GAMB_LIB} ${Boost_INCLUDE_DIRS} )
link_directories(${GAMB_LIB})

set(SOURCE_FILES cnpy.cpp complex_distogram_maker.cpp)
add_executable(complex_distogram_maker ${SOURCE_FILES})
target_link_libraries(complex_distogram_maker ${Boost_LIBRARIES} gamb++ z stdc++fs)
