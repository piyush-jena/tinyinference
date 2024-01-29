## tinyinference

This project aims to create a framework for language model inference using C++ to allow models to run on any device. It was inspired from [llama2.c](https://github.com/karpathy/llama2.c) by @[karpathy](https://github.com/karpathy).

The main objective of this project is to create building blocks using C++ to be able to synthesize models quickly using syntax similar to pytorch. I aim to encapsulate all the complexity inside the library and achieve as high a speed as possible without sacrificing the simplicity of the code too much.

## Planned Tasks
- performance benchmark of llama2 in macbook air
- update llama2 to make use of linear layers instead of tensors
- add unit tests
- add multi-threading wherever possible
