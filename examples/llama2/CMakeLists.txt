add_executable("a.out" "main.cpp")
target_link_libraries("a.out" PRIVATE ${TINYINFERENCE_LIB})