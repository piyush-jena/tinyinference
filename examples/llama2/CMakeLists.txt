add_executable("llama2" "llama2.cpp")
target_link_libraries("llama2" PRIVATE ${TINYINFERENCE_LIB})