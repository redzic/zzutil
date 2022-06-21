# Set the minimum version of CMake that can be used
# To find the cmake version run
# $ cmake --version
cmake_minimum_required(VERSION 3.9.4)

# Set the project name
project(zzutil)

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

# set(CMAKE_C_CLANG_TIDY clang-tidy -checks=*,-altera-*,-llvmlibc-restrict-system-libc-headers,-readability-magic-numbers,-cppcoreguidelines-avoid-magic-numbers,-readability-isolate-declaration)
# set(CMAKE_C_CPPCHECK "cppcheck")
set(SOURCES
    src/wc.c
    src/memcount_avx2.c
)

IF(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set_source_files_properties(src/memcount_avx2.c PROPERTIES COMPILE_FLAGS -mavx2)
ELSEIF(MSVC)
    set_source_files_properties(src/memcount_avx2.c PROPERTIES COMPILE_FLAGS /arch:AVX2)
ENDIF()

# Add an executable
add_executable(wc ${SOURCES})
