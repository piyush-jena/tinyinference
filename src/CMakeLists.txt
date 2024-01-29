add_library(tinyinference-objs OBJECT
	tensor.cpp
	mathlib.cpp
	nn/linear.cpp
	nn/embedding.cpp
	encoder/bpe.cpp
)

set_target_properties(tinyinference-objs PROPERTIES POSITION_INDEPENDENT_CODE 1)

target_include_directories(tinyinference-objs
	PUBLIC
		$<BUILD_INTERFACE:${TINYINFERENCE_INCLUDE_DIR}>
		$<INSTALL_INTERFACE:include>
	PRIVATE 
		${TINYINFERENCE_GENERATED_DIR}/include
		${CMAKE_SOURCE_DIR}/src
)
