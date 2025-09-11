cmake_minimum_required(VERSION 3.10)
project(AnkleSolver VERSION 1.0.0 LANGUAGES C CXX)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 设置 C 标准
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# 编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O2")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -O2")

# 查找 Eigen3
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
if(NOT Eigen3_FOUND)
    message(FATAL_ERROR "Eigen3 library not found! Please install Eigen3.")
endif()

# 创建 CasADi 生成函数的静态库
add_library(ankle_functions STATIC
    autogen_code/ankle_functions.c
)

# 设置 C 编译选项
target_compile_options(ankle_functions PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-std=c99>
)

# 创建 AnkleSolver C++ 包装库
add_library(ankle_solver STATIC
    src/ankle_solver.cpp
    include/ankle_solver.h
)

# 链接库
target_link_libraries(ankle_solver
    ankle_functions
    Eigen3::Eigen
    m  # 数学库
)

# 设置包含目录
target_include_directories(ankle_solver PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# 创建示例可执行文件
add_executable(example
    test/example.cpp
)

# 链接示例程序
target_link_libraries(example
    ankle_solver
    Eigen3::Eigen
)

# 安装目标 (可选)
install(TARGETS ankle_solver ankle_functions
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES ankle_solver.h
    DESTINATION include
)

# 创建自定义目标用于运行示例
add_custom_target(run_example
    COMMAND example
    DEPENDS example
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running ankle solver example"
)

# 打印配置信息
message(STATUS "AnkleSolver Configuration:")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  C Standard: ${CMAKE_C_STANDARD}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Eigen3 Version: ${Eigen3_VERSION}")
message(STATUS "  Install Prefix: ${CMAKE_INSTALL_PREFIX}")
