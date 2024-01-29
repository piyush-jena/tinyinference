# --- CMake required version ---

cmake_minimum_required(VERSION 3.12)

# --- Project setup ---

project(tinyinference)

# --- Build Options ---

option(TINYINFERENCE_BUILD_SHARED "Build shared library" OFF)
option(TINYINFERENCE_BUILD_STATIC "Build static library" ON)
option(TINYINFERENCE_BUILD_EXAMPLES "Build example applications" ON)
option(TINYINFERENCE_BUILD_TESTS "Build unit tests" OFF)

# --- Setting naming variables ---

set(TINYINFERENCE_SHARED_LIBRARY tinyinference)
set(TINYINFERENCE_STATIC_LIBRARY tinyinference-static)
set(TINYINFERENCE_OBJECT_LIBRARY tinyinference-objs)

set(TINYINFERENCE_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)
set(TINYINFERENCE_GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)

# --- Project uses C++17 ---

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- Common library sources, etc ---

add_subdirectory(src)

# --- Collect the targets names ---

if(${TINYINFERENCE_BUILD_SHARED})
	list(APPEND TINYINFERENCE_TARGETS ${TINYINFERENCE_SHARED_LIBRARY})
endif()

if(${TINYINFERENCE_BUILD_STATIC})
	list(APPEND TINYINFERENCE_TARGETS ${TINYINFERENCE_STATIC_LIBRARY})
endif()

# --- Create the libraries and export them ---

if(NOT TINYINFERENCE_TARGETS)
	message(FATAL_ERROR "No targets are specified")
endif()

if(${TINYINFERENCE_BUILD_SHARED})
	message(STATUS "Creating shared library: ${TINYINFERENCE_SHARED_LIBRARY}")
	add_library(${TINYINFERENCE_SHARED_LIBRARY} SHARED $<TARGET_OBJECTS:${TINYINFERENCE_OBJECT_LIBRARY}>)

	target_include_directories(${TINYINFERENCE_SHARED_LIBRARY}
		PUBLIC
			$<BUILD_INTERFACE:${TINYINFERENCE_INCLUDE_DIR}>
			$<INSTALL_INTERFACE:include>
		PRIVATE
			${TINYINFERENCE_GENERATED_DIR}/include
	)

	list(APPEND TARGET_FILES ${TINYINFERENCE_SHARED_LIBRARY})
endif()

if(${TINYINFERENCE_BUILD_STATIC})
	message(STATUS "Creating static library: ${TINYINFERENCE_STATIC_LIBRARY}")
	add_library(${TINYINFERENCE_STATIC_LIBRARY} STATIC $<TARGET_OBJECTS:${TINYINFERENCE_OBJECT_LIBRARY}>)

	target_include_directories(${TINYINFERENCE_STATIC_LIBRARY}
		PUBLIC
			$<BUILD_INTERFACE:${TINYINFERENCE_INCLUDE_DIR}>
			$<INSTALL_INTERFACE:include>
		PRIVATE
			${TINYINFERENCE_GENERATED_DIR}/include
	)

	# On *nix systems, the static library can have the same base filename
	# as the shared library, thus 'libsockpp.a' for the static lib.
	# On Windows they need different names to tell the static lib from the
	# DLL import library.
	if(UNIX)
		set_target_properties(${TINYINFERENCE_STATIC_LIBRARY} PROPERTIES 
			OUTPUT_NAME ${TINYINFERENCE_SHARED_LIBRARY}
		)
	endif()

	list(APPEND TARGET_FILES ${TINYINFERENCE_STATIC_LIBRARY})
endif()

# --- Install Targets ---

include(GNUInstallDirs)

install(TARGETS ${TARGET_FILES}
	EXPORT tinyinference-targets
	ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
	LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
	RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(EXPORT tinyinference-targets
	FILE
		tinyinferenceTargets.cmake
	NAMESPACE
		Tinyinference::
	DESTINATION
		${CMAKE_INSTALL_LIBDIR}/cmake/tinyinference
)

include(CMakePackageConfigHelpers)

install(DIRECTORY include/ ${TINYINFERENCE_GENERATED_DIR}/include/
	DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(
	FILES
		${PROJECT_SOURCE_DIR}/cmake/tinyinferenceConfig.cmake
		${TINYINFERENCE_GENERATED_DIR}/cmake/tinyinferenceConfigVersion.cmake
	DESTINATION
		${CMAKE_INSTALL_LIBDIR}/cmake/tinyinference
)

# --- Default library for examples and unit tests ---

if(TINYINFERENCE_BUILD_SHARED)
	set(TINYINFERENCE_LIB ${TINYINFERENCE_SHARED_LIBRARY})
else()
	set(TINYINFERENCE_LIB ${TINYINFERENCE_STATIC_LIBRARY})
endif()

# --- Example applications ---

if(TINYINFERENCE_BUILD_EXAMPLES)
	add_subdirectory(examples)
endif()

# --- Unit Tests ---

if(TINYINFERENCE_BUILD_TESTS)
	add_subdirectory(tests/unit)
endif()