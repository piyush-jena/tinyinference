add_executable("llama2.out" "llama2.cpp")
target_link_libraries("llama2.out" PRIVATE ${TINYINFERENCE_LIB})