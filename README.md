# c-deep-net
A deep learning library implemented from scratch in ANSI C, including a tensor manipulation library, a library for automatic differentiation, and a library for composing neural nets on top of those.
## Testing Workflow
Compile tests with: `gcc -Wall -pedantic tensor.c tensor_test.c -o tensor_test.out`
Check for memory leaks with: `leaks -atExit -- ./tensor_test.out`
Also use the '-Wconversion' flag to catch bad implicit conversions (necessary especially since adding support for negative strides, which makes it a signed type).
## Progress
1. Basic tensor library
   - Basic operations: Creation, views, indexing, freeing
   - Support for efficient (O(1)) view-based unary ops (transpose, broadcast, flip)
   - Support for non-view-based unary ops: element-wise polynomial, tanh, column sum (this one is 2d only for now)
   - Support for 2d binary ops: matrix multiply, add, elementwise multiply, convolutions
2. Initial implementation of NN library: forward and backward passes for linear layer, tanh layer, and convolutional layer (nit for convolution layer: backward pass for the special case of dilated convolutions not supported yet).
## Nits
1. Testing - make it use expect tests (currently just manual print-based testing)
2. E2E Demo with simple conv net
3. Misc todo's and tbd's in code
4. Clean up documentation
5. Full auto-diff if we need it (not necessary at the moment)
