cmake_minimum_required(VERSION 3.8)
project(InterfaceFeatures)
include(GNUInstallDirs)
set(CMAKE_CXX_STANDARD 17)

include_directories(${Boost_INCLUDE_DIRS} )

set(SOURCE_FILES Atom.cc  Matrix3.cc  PDB.cc  RigidTrans3.cc  Rotation3.cc cnpy.cpp complex_distogram_maker.cpp)
add_executable(complex_distogram_maker ${SOURCE_FILES})
target_link_libraries(complex_distogram_maker ${Boost_LIBRARIES} z stdc++fs)
