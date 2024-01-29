#include "llama2.h"
#include <iostream>

int main() {
    char *path = "model.txt";
    llama2 model{path};
    return 0;
}