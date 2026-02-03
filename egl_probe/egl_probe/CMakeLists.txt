cmake_minimum_required(VERSION 3.10)
project(egl_probe)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(PythonInterp REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(X11 REQUIRED)

include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(.)
include_directories(glad)
include_directories(${X11_INCLUDE_DIR})

# Shared source files (GLAD/EGL loaders)
set(COMMON_SOURCES
    "glad/egl.cpp"
    "glad/gl.cpp"
    "glad/glx_dyn.cpp"
)

# --- Target 1: query_devices ---
add_executable(query_devices "query_devices.cpp" ${COMMON_SOURCES})
target_link_libraries(query_devices ${PYTHON_LIBRARIES})
target_link_libraries(query_devices ${X11_LIBRARIES})
target_link_libraries(query_devices EGL GL dl)

# --- Target 2: test_device ---
add_executable(test_device "test_device.cpp" ${COMMON_SOURCES})
target_link_libraries(test_device ${PYTHON_LIBRARIES})
target_link_libraries(test_device ${X11_LIBRARIES})
target_link_libraries(test_device EGL GL dl)

# Output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
