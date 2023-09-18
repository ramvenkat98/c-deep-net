# c-deep-net
A deep learning library implemented from scratch in ANSI C, including a tensor manipulation library, a library for automatic differentiation, and a library for composing neural nets on top of those.
## Testing Workflow
Compile tests with: `gcc -Wall -pedantic tensor.c tensor_test.c -o tensor_test.out`
Check for memory leaks with: `leaks -atExit -- ./tensor_test.out`
## Progress
1. Basic tensor library (creation, views, indexing, freeing) with correctness and efficiency tests
2. Support for view-based unary ops (transpose, broadcast); support for 2d binary ops (matrix multiply, add); support for 2d->1d unary op (col sum). Correctness tests added.
3. Initial implementation of linear layer, gradients. No need for AD at the moment.
## Nits
1. Testing - make it use expect tests (currently just manual print-based testing)
2. Extending support for `add` beyond 2d (currently we don't need it)
3. Full AD once we need it
4. Misc todo's and tbd's in code (e.g. refactoring the 2d indexing to go through an inline function)
