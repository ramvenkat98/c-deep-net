# c-deep-net
A deep learning library implemented from scratch in ANSI C, including a tensor manipulation library, a library for automatic differentiation, and a library for composing neural nets on top of those.

## Testing Workflow
Compile tests with: `gcc -Wall -pedantic tensor.c tensor_test.c -o tensor_test.out`
Check for memory leaks with: `leaks -atExit -- ./tensor_test.out`
